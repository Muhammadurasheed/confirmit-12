import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import Container from "@/components/layout/Container";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Shield, Zap, Lock, ArrowLeft, Download, Share2 } from "lucide-react";
import { toast } from "sonner";
import { useReceiptStore } from "@/store/receiptStore";
import { useReceiptUpload } from "@/hooks/useReceiptUpload";
import { useFirebaseReceiptProgress } from "@/hooks/useFirebaseReceiptProgress";
import { UploadZone } from "@/components/features/receipt-scan/UploadZone";
import { AnalysisProgress } from "@/components/features/receipt-scan/AnalysisProgress";
import { ResultsDisplay } from "@/components/features/receipt-scan/ResultsDisplay";
import { useAuth } from "@/hooks/useAuth";
import { db } from "@/lib/firebase";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";

const QuickScan = () => {
  const [anchorOnHedera, setAnchorOnHedera] = useState(false);
  const [uploadedReceiptId, setUploadedReceiptId] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const { user } = useAuth();

  const {
    currentReceipt,
    analysisProgress,
    analysisStatus,
    isAnalyzing,
    results,
    setReceipt,
    updateProgress,
    setResults,
    startAnalysis,
    completeAnalysis,
    reset,
  } = useReceiptStore();

  const { uploadReceipt, isUploading } = useReceiptUpload();

  // Save receipt to history after analysis completes
  const saveToHistory = async (receiptData: any, analysisResults: any) => {
    try {
      const historyEntry = {
        receipt_id: receiptData.receiptId,
        user_id: user?.uid || 'anonymous',
        storage_path: receiptData.storagePath,
        trust_score: analysisResults.trust_score || analysisResults.trustScore || 0,
        verdict: analysisResults.verdict || 'unclear',
        merchant_name: analysisResults.merchant?.name || null,
        created_at: serverTimestamp(),
        analysis_data: analysisResults,
      };

      if (user && db) {
        // Save to Firestore for authenticated users
        await addDoc(collection(db, 'receipts'), historyEntry);
        console.log('✅ Receipt saved to Firestore');
      } else {
        // Save to localStorage for anonymous users
        const localHistory = localStorage.getItem('receipt_history');
        const history = localHistory ? JSON.parse(localHistory) : [];
        history.unshift({
          ...historyEntry,
          id: receiptData.receiptId,
          created_at: new Date().toISOString(),
        });
        localStorage.setItem('receipt_history', JSON.stringify(history.slice(0, 50)));
        console.log('✅ Receipt saved to localStorage');
      }
    } catch (error) {
      console.error('❌ Failed to save to history:', error);
      // Don't show error to user - history saving is non-critical
    }
  };

  // Firebase Real-time Progress Listener (CORRECT way - agents emit to Firebase)
  useFirebaseReceiptProgress({
    receiptId: uploadedReceiptId || undefined,
    onProgress: (data) => {
      console.log(`🔥 [${data.agent}] ${data.stage}: ${data.message}`);
      
      // Map agent stages to user-friendly messages
      const statusMap: Record<string, string> = {
        'ocr_started': 'ocr_started',
        'ocr_complete': 'ocr_started',
        'gemini_fallback': 'ocr_started',
        'forensics_running': 'forensics_running',
        'ela_analysis': 'forensics_running',
        'pixel_analysis': 'forensics_running',
        'template_matching': 'forensics_running',
        'analysis_complete': 'analysis_complete',
      };
      
      const mappedStatus = statusMap[data.stage] || 'processing';
      updateProgress(data.progress || 0, mappedStatus, data.message);
      
      // Store agent-specific data for display
      setReceipt({
        ...currentReceipt!,
        currentAgent: data.agent,
        agentDetails: data.details
      });
      
      // Show toast for important milestones
      if (data.stage === 'forensics_running' && data.details?.fraud_indicators) {
        toast.info(`⚠️ ${data.details.fraud_indicators} fraud indicators detected`);
      }
    },
    onComplete: async (analysisData) => {
      console.log('✅ Firebase analysis complete:', analysisData);
      
      console.log('📊 Final analysis data:', JSON.stringify(analysisData, null, 2));
      setResults(analysisData);
      completeAnalysis();
      
      // Save to history
      if (currentReceipt) {
        await saveToHistory(currentReceipt, analysisData);
      }
      
      toast.success('Analysis complete!');
    },
    onError: (error) => {
      console.error('❌ Firebase error in QuickScan:', error);
      toast.error(error.error || error.message || 'Analysis failed');
      useReceiptStore.getState().failAnalysis(error.error || error.message || 'Unknown error');
    },
  });

  const handleFileSelect = async (file: File) => {
    try {
      reset();
      startAnalysis();
      
      const result = await uploadReceipt(file, { anchorOnHedera });
      
      setUploadedReceiptId(result.receiptId);
      setReceipt({
        receiptId: result.receiptId,
        storagePath: result.cloudinaryUrl || '',
        uploadTimestamp: new Date(),
        analysis: null,
        status: 'processing',
      });

      toast.success('Receipt uploaded! Analyzing...');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Upload failed');
      useReceiptStore.getState().failAnalysis('Upload failed');
    }
  };

  const handleReset = () => {
    reset();
    setUploadedReceiptId(null);
  };

  const handleDownloadReport = () => {
    if (!results || !currentReceipt) return;
    
    const report = {
      receiptId: currentReceipt.receiptId,
      timestamp: new Date().toISOString(),
      trustScore: results.trustScore,
      verdict: results.verdict,
      recommendation: results.recommendation,
      issues: results.issues,
      forensicDetails: results.forensicDetails,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `legit-report-${currentReceipt.receiptId}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast.success('Report downloaded!');
  };

  const handleShare = async () => {
    if (!currentReceipt) return;

    const shareUrl = `${window.location.origin}/receipt/${currentReceipt.receiptId}`;
    
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Legit Verification Report',
          text: `Trust Score: ${results?.trustScore}/100`,
          url: shareUrl,
        });
      } catch (error) {
        // User cancelled share
      }
    } else {
      await navigator.clipboard.writeText(shareUrl);
      toast.success('Link copied to clipboard!');
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-gradient-to-br from-background via-background to-muted/20">
      <Header />
      
      <main className="flex-1">
        <Container className="py-8 md:py-12">
          {/* Hero Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <Badge variant="outline" className="mb-4">
              <Shield className="h-3 w-3 mr-1" />
              AI-Powered Verification
            </Badge>
            <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              Receipt Verification
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto">
              Upload any receipt and get instant AI-powered fraud analysis. 
              Know if you're dealing with a legitimate business in seconds.
            </p>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6 mb-8 md:mb-12"
          >
            <Card className="border-primary/20 hover:border-primary/40 transition-colors">
              <CardHeader>
                <Zap className="h-8 w-8 text-primary mb-2" />
                <CardTitle className="text-lg">Instant Analysis</CardTitle>
                <CardDescription className="text-sm">
                  Get results in under 8 seconds with AI-powered forensics
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-primary/20 hover:border-primary/40 transition-colors">
              <CardHeader>
                <Shield className="h-8 w-8 text-green-500 mb-2" />
                <CardTitle className="text-lg">Multi-Agent AI</CardTitle>
                <CardDescription className="text-sm">
                  Advanced computer vision and forensic analysis agents
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-primary/20 hover:border-primary/40 transition-colors">
              <CardHeader>
                <Lock className="h-8 w-8 text-purple-500 mb-2" />
                <CardTitle className="text-lg">Blockchain Proof</CardTitle>
                <CardDescription className="text-sm">
                  Anchor results to Hedera for immutable verification
                </CardDescription>
              </CardHeader>
            </Card>
          </motion.div>

          {/* Main Content */}
          <AnimatePresence mode="wait">
            {!isAnalyzing && !results && (
              <motion.div
                key="upload"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.2 }}
              >
                <Card className="shadow-2xl border-primary/10">
                  <CardHeader className="space-y-4">
                    <div>
                      <CardTitle className="text-2xl">Upload Receipt</CardTitle>
                      <CardDescription className="text-base mt-2">
                        Drag and drop, take a photo, or browse your files
                      </CardDescription>
                    </div>
                    
                    {/* Hedera Anchoring Option */}
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50 border">
                      <div className="flex items-center gap-3">
                        <Lock className="h-5 w-5 text-purple-500" />
                        <div>
                          <Label htmlFor="hedera-anchor" className="font-semibold cursor-pointer">
                            Anchor to Hedera Blockchain
                          </Label>
                          <p className="text-xs text-muted-foreground mt-1">
                            Create immutable proof of verification (recommended)
                          </p>
                        </div>
                      </div>
                      <Switch
                        id="hedera-anchor"
                        checked={anchorOnHedera}
                        onCheckedChange={setAnchorOnHedera}
                      />
                    </div>
                  </CardHeader>
                  
                  <CardContent>
                    <UploadZone
                      onFileSelected={(file) => setSelectedFile(file)}
                      onScanClick={handleFileSelect}
                      disabled={isUploading || isAnalyzing}
                      isScanning={isAnalyzing}
                    />
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {isAnalyzing && (
              <motion.div
                key="analyzing"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
              >
                <AnalysisProgress
                  progress={analysisProgress}
                  status={analysisStatus}
                  message={`Receipt ID: ${currentReceipt?.receiptId || ''}`}
                />
              </motion.div>
            )}

            {results && currentReceipt && (
              <motion.div
                key="results"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
              >
                {/* Action Buttons */}
                <div className="flex flex-wrap gap-3 mb-6">
                  <Button onClick={handleReset} variant="outline" size="lg">
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Scan Another
                  </Button>
                  <Button onClick={handleDownloadReport} variant="outline" size="lg">
                    <Download className="h-4 w-4 mr-2" />
                    Download Report
                  </Button>
                  <Button onClick={handleShare} variant="outline" size="lg">
                    <Share2 className="h-4 w-4 mr-2" />
                    Share
                  </Button>
                </div>

                {/* Results */}
                <ResultsDisplay
                  receiptId={currentReceipt.receiptId}
                  receiptImageUrl={currentReceipt.storagePath}
                  trustScore={(results.trust_score || results.trustScore || 0) as number}
                  verdict={(results.verdict || 'unclear') as any}
                  issues={results.issues || []}
                  recommendation={results.recommendation || 'Analysis completed. Review the details below.'}
                  forensicDetails={{
                    ocr_confidence: (results.forensic_details?.ocr_confidence || results.forensicDetails?.ocr_confidence || results.forensicDetails?.ocrConfidence || 0) as number,
                    manipulation_score: (results.forensic_details?.manipulation_score || results.forensicDetails?.manipulation_score || results.forensicDetails?.manipulationScore || 0) as number,
                    metadata_flags: (results.forensic_details?.metadata_flags || results.forensicDetails?.metadata_flags || results.forensicDetails?.metadataFlags || []) as string[],
                    technical_details: results.forensic_details?.technical_details || results.forensicDetails?.technical_details,
                  }}
                  merchant={results.merchant ? {
                    name: results.merchant.name,
                    verified: results.merchant.verified,
                    trust_score: (results.merchant.trust_score || results.merchant.trustScore || 0) as number,
                  } : undefined}
                  hederaAnchor={currentReceipt.hederaAnchor ? {
                    transaction_id: currentReceipt.hederaAnchor.transactionId,
                    consensus_timestamp: currentReceipt.hederaAnchor.consensusTimestamp,
                    explorer_url: currentReceipt.hederaAnchor.explorerUrl,
                  } : undefined}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </Container>
      </main>
      
      <Footer />
    </div>
  );
};

export default QuickScan;
