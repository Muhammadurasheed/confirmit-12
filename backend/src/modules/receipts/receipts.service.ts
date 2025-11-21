import { Injectable, Logger, Inject } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as admin from 'firebase-admin';
import { v2 as cloudinary } from 'cloudinary';
import { ScanReceiptDto } from './dto/scan-receipt.dto';
import { ReceiptsGateway } from './receipts.gateway';
import { HederaService } from '../hedera/hedera.service';
import axios from 'axios';

@Injectable()
export class ReceiptsService {
  private readonly logger = new Logger(ReceiptsService.name);

  constructor(
    private readonly configService: ConfigService,
    @Inject('FIRESTORE') private readonly db: admin.firestore.Firestore,
    private readonly receiptsGateway: ReceiptsGateway,
    private readonly hederaService: HederaService,
  ) {
    // Initialize Cloudinary with environment variables from ConfigService
    const cloudinaryConfig = {
      cloud_name: this.configService.get<string>('cloudinary.cloudName') || this.configService.get<string>('CLOUDINARY_CLOUD_NAME'),
      api_key: this.configService.get<string>('cloudinary.apiKey') || this.configService.get<string>('CLOUDINARY_API_KEY'),
      api_secret: this.configService.get<string>('cloudinary.apiSecret') || this.configService.get<string>('CLOUDINARY_API_SECRET'),
    };

    this.logger.log(`🔧 Cloudinary config: ${cloudinaryConfig.cloud_name} / ${cloudinaryConfig.api_key ? '✅ API key present' : '❌ API key missing'}`);

    cloudinary.config(cloudinaryConfig);
  }

  async scanReceipt(file: Express.Multer.File, dto: ScanReceiptDto) {
    const receiptId = this.generateReceiptId();
    this.logger.log(`Starting receipt scan: ${receiptId}`);

    try {
      // 1. Emit initial progress before starting upload
      this.receiptsGateway.emitProgress(receiptId, 5, 'uploading', 'Starting upload...');
      
      // 2. Upload to Cloudinary
      const uploadResult = await this.uploadToCloudinary(file);
      this.receiptsGateway.emitProgress(receiptId, 15, 'upload_complete', 'Image uploaded successfully');

      // 3. Create receipt document
      await this.db.collection('receipts').doc(receiptId).set({
        receipt_id: receiptId,
        user_id: dto.userId || 'anonymous',
        storage_path: uploadResult.secure_url,
        cloudinary_public_id: uploadResult.public_id,
        upload_timestamp: admin.firestore.FieldValue.serverTimestamp(),
        status: 'processing',
        analysis: null,
        hedera_anchor: null,
      });

      // 4. Start async analysis
      this.analyzeReceiptAsync(receiptId, uploadResult.secure_url, dto.anchorOnHedera);

      return {
        success: true,
        receiptId,
        message: 'Receipt scan initiated',
      };
    } catch (error) {
      this.logger.error(`Receipt scan failed: ${error.message}`, error.stack);
      throw error;
    }
  }

  private async analyzeReceiptAsync(
    receiptId: string,
    imageUrl: string,
    anchorOnHedera: boolean,
  ) {
    const receiptRef = this.db.collection('receipts').doc(receiptId);
    const startTime = Date.now();

    try {
      // Send progress updates
      this.receiptsGateway.emitProgress(receiptId, 20, 'ocr_started', 'Extracting text with AI...');

      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate OCR

      this.receiptsGateway.emitProgress(receiptId, 40, 'forensics_running', 'Running forensic analysis...');

      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate forensics

      // Call AI service with proper error handling
      const aiServiceUrl = this.configService.get('aiService.url');
      this.logger.log(`Calling AI service: ${aiServiceUrl}/api/analyze-receipt`);

      let aiResponse;
      let analysisResult;

      try {
        aiResponse = await axios.post(
          `${aiServiceUrl}/api/analyze-receipt`,
          {
            image_url: imageUrl,
            receipt_id: receiptId,
          },
          {
            timeout: 180000, // 3 minute timeout for Gemini + forensics
            headers: {
              'Content-Type': 'application/json',
            },
          },
        );

        analysisResult = aiResponse.data;
      } catch (aiError) {
        this.logger.error(`AI service error: ${aiError.message}`, aiError.stack);

        // Check if it's a timeout
        if (aiError.code === 'ECONNABORTED' || aiError.message.includes('timeout')) {
          throw new Error(
            'AI analysis timed out. The image might be too large or the service is busy. Please try again.',
          );
        }

        // Check if service is down
        if (aiError.code === 'ECONNREFUSED' || aiError.code === 'ENOTFOUND') {
          throw new Error(
            'AI service is currently unavailable. Our team has been notified. Please try again in a few minutes.',
          );
        }

        // Generic AI error
        throw new Error(
          aiError.response?.data?.detail || 
          aiError.response?.data?.message ||
          'AI analysis failed. Please try again with a clearer image.',
        );
      }

      this.receiptsGateway.emitProgress(receiptId, 80, 'analysis_complete', 'Analysis complete!');

      // Store complete results with ALL data from AI service
      await receiptRef.update({
        ocr_text: analysisResult.ocr_text || '',  // CRITICAL: Store OCR text at root level
        analysis: {
          trust_score: analysisResult.trust_score,
          verdict: analysisResult.verdict,
          issues: analysisResult.issues || [],
          recommendation: analysisResult.recommendation,
          forensic_details: {
            ocr_confidence: analysisResult.forensic_details?.ocr_confidence || 0,
            manipulation_score: analysisResult.forensic_details?.manipulation_score || 0,
            metadata_flags: analysisResult.forensic_details?.metadata_flags || [],
            forensic_summary: analysisResult.forensic_details?.forensic_summary,
            techniques_detected: analysisResult.forensic_details?.techniques_detected || [],
            authenticity_indicators: analysisResult.forensic_details?.authenticity_indicators || [],
            forensic_progress: analysisResult.forensic_details?.forensic_progress || [],
            technical_details: analysisResult.forensic_details?.technical_details || {},
            heatmap: analysisResult.forensic_details?.heatmap || [],  // ELA heatmap data
            pixel_diff: analysisResult.forensic_details?.pixel_diff || null,  // Pixel diff map
            ela_analysis: {
              manipulation_detected: analysisResult.forensic_details?.manipulation_detected || false,
              heatmap: analysisResult.forensic_details?.heatmap || [],
              suspicious_regions: analysisResult.forensic_details?.suspicious_regions || [],
              image_dimensions: analysisResult.forensic_details?.image_dimensions,
              statistics: analysisResult.forensic_details?.statistics,
              techniques: analysisResult.forensic_details?.techniques_detected || [],
              pixel_diff: analysisResult.forensic_details?.pixel_diff || null,
            },
          },
          merchant: analysisResult.merchant || null,
          agent_logs: analysisResult.agent_logs || [],
        },
        processing_time: Date.now() - startTime,
        status: 'completed',
      });

      // Anchor to Hedera if requested
      if (anchorOnHedera) {
        this.receiptsGateway.emitProgress(receiptId, 90, 'hedera_anchoring', 'Anchoring to blockchain...');

        const hederaResult = await this.hederaService.anchorToHCS(receiptId, analysisResult);

        await receiptRef.update({
          hedera_anchor: hederaResult,
        });

        this.receiptsGateway.emitProgress(receiptId, 100, 'hedera_anchored', 'Verified on blockchain!');
      } else {
        this.receiptsGateway.emitProgress(receiptId, 100, 'complete', 'Verification complete!');
      }

      // Send final results
      const finalDoc = await receiptRef.get();
      this.receiptsGateway.emitComplete(receiptId, finalDoc.data());

      this.logger.log(`Receipt analysis completed: ${receiptId} in ${Date.now() - startTime}ms`);
    } catch (error) {
      this.logger.error(`Receipt analysis failed for ${receiptId}: ${error.message}`, error.stack);

      const userFriendlyError = this.getUserFriendlyError(error);

      await receiptRef.update({
        status: 'failed',
        error: userFriendlyError,
        error_timestamp: admin.firestore.FieldValue.serverTimestamp(),
      });

      this.receiptsGateway.emitError(receiptId, userFriendlyError);
    }
  }

  private getUserFriendlyError(error: any): string {
    // Map technical errors to user-friendly messages
    const message = error.message || 'Unknown error';

    if (message.includes('timeout')) {
      return 'Analysis timed out. The image might be too large. Please try a smaller image.';
    }
    if (message.includes('unavailable') || message.includes('ECONNREFUSED')) {
      return 'Service temporarily unavailable. Please try again in a few minutes.';
    }
    if (message.includes('invalid image') || message.includes('format')) {
      return 'Invalid image format. Please upload a JPG, PNG, or PDF file.';
    }
    if (message.includes('too large')) {
      return 'Image file is too large. Please upload a file smaller than 10MB.';
    }

    return message;
  }

  async getReceipt(receiptId: string) {
    const doc = await this.db.collection('receipts').doc(receiptId).get();

    if (!doc.exists) {
      throw new Error('Receipt not found');
    }

    return {
      success: true,
      data: doc.data(),
    };
  }

  async getUserReceipts(userId: string) {
    const snapshot = await this.db
      .collection('receipts')
      .where('user_id', '==', userId)
      .orderBy('upload_timestamp', 'desc')
      .limit(50)
      .get();

    const receipts = snapshot.docs.map((doc) => doc.data());

    return {
      success: true,
      count: receipts.length,
      data: receipts,
    };
  }

  private async uploadToCloudinary(file: Express.Multer.File): Promise<any> {
    return new Promise((resolve, reject) => {
      const uploadStream = cloudinary.uploader.upload_stream(
        {
          folder: 'confirmit/receipts',
          resource_type: 'image',
          transformation: [{ quality: 'auto:best' }, { fetch_format: 'auto' }],
        },
        (error, result) => {
          if (error) reject(error);
          else resolve(result);
        },
      );

      uploadStream.end(file.buffer);
    });
  }

  async anchorReceiptToHedera(receiptId: string) {
    this.logger.log(`Anchoring receipt ${receiptId} to Hedera`);

    try {
      const receiptRef = this.db.collection('receipts').doc(receiptId);
      const doc = await receiptRef.get();

      if (!doc.exists) {
        throw new Error('Receipt not found');
      }

      const receiptData = doc.data();

      if (receiptData.hedera_anchor) {
        // Already anchored
        return {
          success: true,
          message: 'Receipt already anchored',
          hedera_anchor: receiptData.hedera_anchor,
        };
      }

      // Anchor to Hedera
      const hederaResult = await this.hederaService.anchorToHCS(
        receiptId,
        receiptData.analysis,
      );

      await receiptRef.update({
        hedera_anchor: hederaResult,
      });

      return {
        success: true,
        message: 'Successfully anchored to Hedera',
        hedera_anchor: hederaResult,
      };
    } catch (error) {
      this.logger.error(`Hedera anchoring failed: ${error.message}`, error.stack);
      throw error;
    }
  }

  private generateReceiptId(): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 9);
    return `RCP-${timestamp}${random}`.toUpperCase();
  }
}
