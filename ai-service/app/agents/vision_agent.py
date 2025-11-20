"""
Vision Agent - Uses Tesseract OCR (primary) + Gemini Vision (fallback) for receipt analysis
"""
import logging
import google.generativeai as genai
from PIL import Image
from typing import Dict, Any
import pytesseract
import re
import asyncio

logger = logging.getLogger(__name__)


class VisionAgent:
    """Gemini Vision API wrapper for receipt OCR and visual analysis"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Use gemini-1.5-flash - more stable, better quotas than experimental model
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def analyze(self, image_path: str, progress=None) -> Dict[str, Any]:
        """
        Analyze receipt image using Tesseract OCR (primary) + Gemini Vision (fallback)

        Returns:
            - ocr_text: Extracted text from receipt
            - confidence: OCR confidence score (0-100)
            - visual_anomalies: List of detected visual issues
            - merchant_name: Detected merchant name
            - total_amount: Detected total amount
            - receipt_date: Detected date
        """
        try:
            logger.info(f"Vision agent analyzing: {image_path}")
            
            if progress:
                await progress.emit(
                    agent="vision",
                    stage="ocr_started",
                    message="Extracting text from receipt using AI OCR",
                    progress=25
                )

            # Load image
            img = Image.open(image_path)
            
            # STAGE 1: Try Tesseract OCR first (FREE, fast, local)
            tesseract_result = await self._analyze_with_tesseract(img)
            
            # If Tesseract confidence is good, use it
            if tesseract_result['confidence'] >= 70:
                logger.info(f"✅ Tesseract OCR successful with {tesseract_result['confidence']}% confidence")
                
                if progress:
                    await progress.emit(
                        agent="vision",
                        stage="ocr_complete",
                        message=f"Extracted text from {tesseract_result.get('merchant_name', 'receipt')}",
                        progress=35,
                        details={
                            'merchant': tesseract_result.get('merchant_name'),
                            'amount': tesseract_result.get('total_amount'),
                            'confidence': tesseract_result['confidence']
                        }
                    )
                
                return tesseract_result
            
            # STAGE 2: Fallback to Gemini if Tesseract confidence is low
            logger.warning(f"⚠️ Tesseract confidence low ({tesseract_result['confidence']}%), falling back to Gemini")
            
            if progress:
                await progress.emit(
                    agent="vision",
                    stage="gemini_fallback",
                    message="Using advanced AI vision for better accuracy",
                    progress=30
                )
            
            gemini_result = await self._analyze_with_gemini(img, progress)
            return gemini_result
            
        except Exception as e:
            logger.error(f"Vision agent error: {str(e)}")
            raise

    async def _analyze_with_tesseract(self, img: Image) -> Dict[str, Any]:
        """Use Tesseract OCR for text extraction (FREE, local)"""
        try:
            # Extract text with confidence data
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Get full text
            ocr_text = pytesseract.image_to_string(img)
            
            # Calculate average confidence (Tesseract returns confidence per word)
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract structured data
            merchant_name = self._extract_merchant(ocr_text)
            total_amount = self._extract_amount(ocr_text)
            receipt_date = self._extract_date(ocr_text)
            account_numbers = self._extract_accounts(ocr_text)
            phone_numbers = self._extract_phones(ocr_text)
            
            return {
                "ocr_text": ocr_text,
                "confidence": int(avg_confidence),
                "merchant_name": merchant_name,
                "total_amount": total_amount,
                "currency": "NGN" if total_amount else None,
                "receipt_date": receipt_date,
                "items": [],
                "account_numbers": account_numbers,
                "phone_numbers": phone_numbers,
                "visual_quality": "excellent" if avg_confidence >= 80 else "good",
                "visual_anomalies": [],
                "ocr_method": "tesseract"
            }
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {str(e)}")
            return {"confidence": 0, "ocr_text": "", "ocr_method": "tesseract_failed"}
    
    def _extract_merchant(self, text: str) -> str:
        """Extract merchant name from OCR text"""
        lines = text.split('\n')
        # Merchant name usually in first few lines, capitalized
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 3 and line.isupper():
                return line
        return None
    
    def _extract_amount(self, text: str) -> float:
        """Extract total amount from OCR text"""
        # Match patterns like: ₦1,500.00, N1500, NGN 1,500
        patterns = [
            r'(?:₦|N|NGN)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:AMOUNT|TOTAL|PAID)[\s:]+(?:₦|N|NGN)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        return None
    
    def _extract_date(self, text: str) -> str:
        """Extract date from OCR text"""
        # Match patterns: 2024-01-15, 15/01/2024, Jan 15, 2024
        patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def _extract_accounts(self, text: str) -> list:
        """Extract account numbers (10 digits for Nigerian banks)"""
        return re.findall(r'\b\d{10}\b', text)
    
    def _extract_phones(self, text: str) -> list:
        """Extract phone numbers"""
        return re.findall(r'\b(?:\+234|0)\d{10}\b', text)

    async def _analyze_with_gemini(self, img: Image, image_path: str, progress=None) -> Dict[str, Any]:
        """Enhanced Gemini Vision API with forensic fraud detection"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔍 Gemini Vision attempt {attempt + 1}/{max_retries}")
                
                if progress:
                    await progress.emit(
                        agent="vision",
                        stage="analyzing",
                        message="Initializing Google Gemini Vision AI for forensic receipt analysis",
                        progress=25
                    )

                # FORENSIC PROMPT - designed to detect manipulated receipts
                prompt = """You are a FORENSIC DOCUMENT ANALYST. Analyze this receipt for authenticity and fraud indicators.

TASK 1: Extract ALL visible text (be thorough - extract everything you can see)

TASK 2: Forensic fraud detection - look for these RED FLAGS:
- Font inconsistencies (weight, size, style variations in similar text fields)
- Color anomalies (same color appearing in different shades - e.g., amount text brighter than surrounding text)
- Spacing issues (unnatural gaps, misaligned text that doesn't match the overall layout)
- Overlay artifacts (text that appears to be added on top of the image)
- Excessive watermarking (suspicious patterns that may hide manipulation)

Return ONLY raw JSON (no markdown, no code blocks, no explanations):
{
  "ocr_text": "Complete extracted text - include everything visible",
  "confidence_score": 85,
  "merchant_name": "Exact business name",
  "total_amount": "Amount value",
  "currency": "NGN",
  "receipt_date": "2025-11-20",
  "items": ["item 1", "item 2"],
  "account_numbers": ["account numbers"],
  "phone_numbers": ["phone numbers"],
  "visual_quality": "excellent|good|fair|poor",
  "visual_anomalies": [
    "Font weight inconsistency detected in amount field - text appears thicker/bolder than surrounding text",
    "Color saturation difference - amount text shows brighter/different shade compared to other text",
    "Unnatural spacing around specific fields - suggests editing",
    "Text overlay artifacts visible - signs of photo manipulation"
  ]
}

CRITICAL: Be thorough with fraud detection. List EVERY suspicious visual pattern you detect."""

                if progress:
                    await progress.emit(
                        agent="vision",
                        stage="gemini_analyzing",
                        message="Gemini AI performing forensic analysis on receipt image",
                        progress=30
                    )

                # Call Gemini with generation config
                response = await self.model.generate_content_async(
                    [prompt, img],
                    generation_config={
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "max_output_tokens": 2048,
                    }
                )
                
                if not response or not response.text:
                    raise Exception("Gemini returned empty response")
                
                response_text = response.text.strip()
                logger.info(f"📄 Gemini response received: {len(response_text)} chars")
                
                # Parse JSON response
                import json
                import re

                # Clean markdown formatting
                cleaned = response_text
                cleaned = re.sub(r'^```json\s*', '', cleaned)
                cleaned = re.sub(r'^```\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
                cleaned = cleaned.strip()

                # Extract JSON
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                
                if not json_match:
                    logger.error(f"❌ No JSON found in Gemini response")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    raise Exception("Failed to parse Gemini response as JSON")
                
                try:
                    analysis_data = json.loads(json_match.group())
                    logger.info(f"✅ Successfully parsed Gemini JSON")
                except json.JSONDecodeError as je:
                    logger.error(f"❌ JSON decode error: {str(je)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    raise

                # Validate and process results
                ocr_text = analysis_data.get("ocr_text", "").strip()
                ocr_length = len(ocr_text)
                
                # CRITICAL: Low text extraction is a RED FLAG
                if ocr_length < 20:
                    logger.warning(f"⚠️ CRITICAL: Only {ocr_length} chars extracted - possible tampering")
                    analysis_data["confidence_score"] = max(0, analysis_data.get("confidence_score", 50) - 40)
                    analysis_data.setdefault("visual_anomalies", []).append(
                        f"Very low text extraction ({ocr_length} chars) - image may be tampered, corrupted, or illegible"
                    )
                
                confidence = max(0, min(100, analysis_data.get("confidence_score", 50)))
                visual_anomalies = analysis_data.get("visual_anomalies", [])
                
                if progress:
                    anomaly_count = len(visual_anomalies)
                    status_msg = f"⚠️ {anomaly_count} fraud indicators detected" if anomaly_count > 0 else "✅ No fraud indicators detected"
                    await progress.emit(
                        agent="vision",
                        stage="analysis_complete",
                        message=f"Vision analysis complete. {status_msg}",
                        progress=40,
                        details={
                            "merchant": analysis_data.get("merchant_name"),
                            "amount": analysis_data.get("total_amount"),
                            "confidence": confidence,
                            "fraud_indicators": anomaly_count
                        }
                    )
                
                result = {
                    "ocr_text": ocr_text,
                    "confidence": confidence,
                    "merchant_name": analysis_data.get("merchant_name"),
                    "total_amount": analysis_data.get("total_amount"),
                    "currency": analysis_data.get("currency"),
                    "receipt_date": analysis_data.get("receipt_date"),
                    "items": analysis_data.get("items", []),
                    "account_numbers": analysis_data.get("account_numbers", []),
                    "phone_numbers": analysis_data.get("phone_numbers", []),
                    "visual_quality": analysis_data.get("visual_quality", "fair"),
                    "visual_anomalies": visual_anomalies,
                    "ocr_method": "gemini"
                }

                logger.info(f"✅ Gemini analysis complete - Confidence: {confidence}%, Anomalies: {len(visual_anomalies)}")
                return result
                
            except Exception as e:
                logger.error(f"❌ Gemini attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"❌ All Gemini attempts failed")
                    raise
        
        # Should never reach here
        raise Exception("Gemini analysis failed after all retries")
                
                # Check if it's a quota/rate limit error
                if '429' in error_msg or 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"⏳ Rate limit hit, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ Gemini Vision failed after {max_retries} attempts due to rate limits")
                        # Return empty result instead of crashing
                        return {
                            "ocr_text": "",
                            "confidence": 0,
                            "merchant_name": None,
                            "total_amount": None,
                            "currency": None,
                            "receipt_date": None,
                            "items": [],
                            "account_numbers": [],
                            "phone_numbers": [],
                            "visual_quality": "poor",
                            "visual_anomalies": ["API quota exceeded"],
                            "ocr_method": "gemini_failed"
                        }
                else:
                    # Non-rate-limit error, raise it
                    logger.error(f"Gemini Vision error: {error_msg}")
                    raise
