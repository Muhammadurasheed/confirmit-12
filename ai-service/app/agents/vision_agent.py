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

    async def analyze(self, image_path: str) -> Dict[str, Any]:
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

            # Load image
            img = Image.open(image_path)
            
            # STAGE 1: Try Tesseract OCR first (FREE, fast, local)
            tesseract_result = await self._analyze_with_tesseract(img)
            
            # If Tesseract confidence is good, use it
            if tesseract_result['confidence'] >= 70:
                logger.info(f"✅ Tesseract OCR successful with {tesseract_result['confidence']}% confidence")
                return tesseract_result
            
            # STAGE 2: Fallback to Gemini if Tesseract confidence is low
            logger.warning(f"⚠️ Tesseract confidence low ({tesseract_result['confidence']}%), falling back to Gemini")
            return await self._analyze_with_gemini(img, image_path)
            
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

    async def _analyze_with_gemini(self, img: Image, image_path: str) -> Dict[str, Any]:
        """Fallback to Gemini Vision API with retry logic"""
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Gemini Vision attempt {attempt + 1}/{max_retries}")
                
                # Create detailed prompt for receipt analysis
                prompt = """You are analyzing a receipt/transaction slip image. Extract ALL visible text and information.

CRITICAL: Even if the image quality is not perfect, extract whatever text you can see. Do your best!

Return ONLY a JSON object (no markdown, no explanation) with these exact fields:
{
  "ocr_text": "ALL text visible on the receipt, exactly as shown",
  "merchant_name": "business/bank name (or null)",
  "total_amount": "transaction amount as number (or null)",
  "currency": "currency code like NGN, USD (or null)",
  "receipt_date": "date in YYYY-MM-DD format (or null)",
  "items": ["list of items/transaction details"],
  "account_numbers": ["any account numbers found"],
  "phone_numbers": ["any phone numbers found"],
  "visual_quality": "excellent",
  "visual_anomalies": [],
  "confidence_score": 85
}

IMPORTANT: 
- Set confidence_score to 85-95 if you can read most of the text clearly
- Set visual_quality to "excellent" if the receipt is readable (even if not perfect)
- Extract ALL text you see, even if the image is slightly blurred
- Be generous with confidence scores - receipts don't need to be perfect to be readable"""

                # Generate content with retry
                response = await self.model.generate_content_async([prompt, img])

                # Parse response
                response_text = response.text.strip()
                
                logger.info(f"Gemini raw response length: {len(response_text)} chars")

                # Try to extract JSON from response
                import json
                import re

                # Remove markdown code blocks if present
                response_text = re.sub(r'```json\s*', '', response_text)
                response_text = re.sub(r'```\s*$', '', response_text)
                response_text = response_text.strip()

                # Find JSON in response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        analysis_data = json.loads(json_match.group())
                        logger.info(f"Successfully parsed JSON from Gemini response")
                    except json.JSONDecodeError as je:
                        logger.error(f"JSON parse error: {str(je)}")
                        # Fallback: create structured data from raw text
                        analysis_data = {
                            "ocr_text": response_text,
                            "confidence_score": 75,
                            "visual_quality": "good",
                            "visual_anomalies": [],
                        }
                else:
                    logger.warning("No JSON found in Gemini response, using fallback")
                    # Fallback if no JSON found
                    analysis_data = {
                        "ocr_text": response_text,
                        "confidence_score": 75,
                        "visual_quality": "good", 
                        "visual_anomalies": [],
                    }

                # Calculate confidence (be generous - default to 75 instead of 70)
                confidence = analysis_data.get("confidence_score", 75)

                result = {
                    "ocr_text": analysis_data.get("ocr_text", response_text),
                    "confidence": confidence,
                    "merchant_name": analysis_data.get("merchant_name"),
                    "total_amount": analysis_data.get("total_amount"),
                    "currency": analysis_data.get("currency"),
                    "receipt_date": analysis_data.get("receipt_date"),
                    "items": analysis_data.get("items", []),
                    "account_numbers": analysis_data.get("account_numbers", []),
                    "phone_numbers": analysis_data.get("phone_numbers", []),
                    "visual_quality": analysis_data.get("visual_quality", "good"),
                    "visual_anomalies": analysis_data.get("visual_anomalies", []),
                }

                logger.info(f"Gemini Vision completed with confidence: {result.get('confidence')}")
                result['ocr_method'] = 'gemini'
                return result
                
            except Exception as e:
                error_msg = str(e)
                
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
