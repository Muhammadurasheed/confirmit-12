"""
Reasoning Agent - Synthesizes all agent outputs into final verdict
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """Synthesize multi-agent results into coherent verdict and recommendation"""

    def __init__(self):
        pass

    async def synthesize(self, agent_results: Dict[str, Any], progress=None) -> Dict[str, Any]:
        """
        Combine results from all agents into final analysis

        Returns:
            - trust_score: 0-100
            - verdict: authentic|suspicious|fraudulent|unclear
            - issues: List of detected issues
            - recommendation: Action recommendation for user
        """
        try:
            logger.info("Reasoning agent synthesizing results")
            
            if progress:
                await progress.emit(
                    agent="reasoning",
                    stage="synthesis_started",
                    message="Synthesizing all agent findings into final verdict",
                    progress=85
                )

            vision_data = agent_results.get("vision", {})
            forensic_data = agent_results.get("forensic", {})
            metadata_data = agent_results.get("metadata", {})
            reputation_data = agent_results.get("reputation", {})

            # Calculate trust score (0-100)
            trust_score = await self._calculate_trust_score(
                vision_data, forensic_data, metadata_data, reputation_data
            )

            # Determine verdict
            verdict = self._determine_verdict(
                trust_score, forensic_data, reputation_data
            )

            # Compile issues
            issues = self._compile_issues(
                vision_data, forensic_data, metadata_data, reputation_data
            )

            # Generate recommendation
            recommendation = self._generate_recommendation(verdict, trust_score, issues)

            result = {
                "trust_score": trust_score,
                "verdict": verdict,
                "issues": issues,
                "recommendation": recommendation,
            }

            logger.info(
                f"Reasoning agent completed. Verdict: {verdict}, Score: {trust_score}"
            )
            
            if progress:
                verdict_messages = {
                    'authentic': 'Receipt appears authentic',
                    'suspicious': 'Receipt has suspicious elements',
                    'fraudulent': 'Receipt appears fraudulent',
                    'unclear': 'Unable to determine authenticity'
                }
                await progress.emit(
                    agent="reasoning",
                    stage="synthesis_complete",
                    message=f"{verdict_messages.get(verdict, 'Analysis complete')} - Trust Score: {trust_score}%",
                    progress=95,
                    details={
                        'trust_score': trust_score,
                        'verdict': verdict,
                        'issues_count': len(issues)
                    }
                )
            
            return result

        except Exception as e:
            logger.error(f"Reasoning agent error: {str(e)}")
            return {
                "trust_score": 50,
                "verdict": "unclear",
                "issues": [
                    {
                        "type": "analysis_error",
                        "severity": "medium",
                        "description": "Unable to complete full analysis",
                    }
                ],
                "recommendation": "Manual verification recommended",
            }

    async def _calculate_trust_score(
        self,
        vision: Dict,
        forensic: Dict,
        metadata: Dict,
        reputation: Dict,
    ) -> int:
        """Calculate weighted trust score from all agents"""
        
        # Start with higher base score (75 instead of 70)
        score = 75

        # Vision confidence (weight: 25%) - more weight on successful OCR
        ocr_confidence = vision.get("confidence", 75) if vision else 75
        # Be more generous with vision confidence
        score += (ocr_confidence - 75) * 0.25

        # Forensic analysis (weight: 25%) - reduced weight
        manipulation_score = forensic.get("manipulation_score", 0) if forensic else 0
        score -= manipulation_score * 0.25

        # Metadata (weight: 15%) - reduced weight, less harsh penalty
        metadata_flags = len(metadata.get("flags", [])) if metadata else 0
        score -= metadata_flags * 3  # Reduced from 5 to 3

        # Reputation (weight: 35%)
        fraud_reports = reputation.get("total_fraud_reports", 0) if reputation else 0
        score -= fraud_reports * 10

        # Bonus for verified merchant
        merchant = reputation.get("merchant") if reputation else None
        if merchant and isinstance(merchant, dict) and merchant.get("verified"):
            score += 15

        # If we successfully extracted text, give a bonus
        if vision and vision.get("ocr_text") and len(vision.get("ocr_text", "")) > 20:
            score += 5  # Bonus for successful text extraction

        # Clamp to 0-100
        return max(0, min(100, int(score)))

    def _determine_verdict(
        self, trust_score: int, forensic: Dict, reputation: Dict
    ) -> str:
        """Determine final verdict based on score and critical findings"""
        
        # Critical red flags
        fraud_reports = reputation.get("total_fraud_reports", 0) if reputation else 0
        manipulation_score = forensic.get("manipulation_score", 0) if forensic else 0

        if fraud_reports >= 3 or manipulation_score >= 80:
            return "fraudulent"
        elif trust_score >= 70:  # Lowered from 75
            return "authentic"
        elif trust_score >= 50:
            return "suspicious"
        elif trust_score >= 25:  # Lowered from 30
            return "unclear"
        else:
            return "fraudulent"

    def _compile_issues(
        self,
        vision: Dict,
        forensic: Dict,
        metadata: Dict,
        reputation: Dict,
    ) -> List[Dict]:
        """Compile all detected issues"""
        issues = []

        # Vision issues
        visual_anomalies = vision.get("visual_anomalies", [])
        for anomaly in visual_anomalies:
            issues.append({
                "type": "visual_anomaly",
                "severity": "medium",
                "description": anomaly,
            })

        # Only flag truly poor quality (lowered threshold from 50 to 40)
        if vision.get("confidence", 100) < 40:
            issues.append({
                "type": "poor_image_quality",
                "severity": "medium",
                "description": "Image quality is poor, reducing confidence",
            })

        # Forensic issues
        techniques = forensic.get("techniques_detected", [])
        for technique in techniques:
            issues.append({
                "type": "forensic_finding",
                "severity": "high",
                "description": f"Detected: {technique}",
            })

        # Metadata issues
        metadata_flags = metadata.get("flags", [])
        for flag in metadata_flags:
            issues.append({
                "type": "metadata_issue",
                "severity": "medium",
                "description": flag,
            })

        # Reputation issues
        fraud_reports = reputation.get("total_fraud_reports", 0)
        if fraud_reports > 0:
            issues.append({
                "type": "fraud_history",
                "severity": "high",
                "description": f"Account has {fraud_reports} verified fraud report(s)",
            })

        return issues

    def _generate_recommendation(
        self, verdict: str, trust_score: int, issues: List[Dict]
    ) -> str:
        """Generate actionable recommendation"""
        
        if verdict == "fraudulent":
            return "🚨 DO NOT PROCEED - This receipt shows clear signs of fraud. Report this merchant immediately."
        elif verdict == "suspicious":
            return "⚠️ CAUTION ADVISED - This receipt has suspicious elements. Verify with merchant directly before proceeding."
        elif verdict == "unclear":
            if len(issues) == 0:
                return "ℹ️ UNCLEAR - Unable to fully verify. Request additional documentation."
            else:
                return f"ℹ️ UNCLEAR - {len(issues)} issue(s) detected. Manual verification recommended."
        else:  # authentic
            if trust_score >= 90:
                return "✅ HIGHLY TRUSTWORTHY - This receipt appears completely authentic."
            else:
                return "✅ LIKELY AUTHENTIC - This receipt appears genuine with minor concerns."
