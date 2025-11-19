# ConfirmIT: Technical Architecture & Review Document

**Project Status:** Post-Hedera Hackathon → Pivoting to Codematic Hackathon  
**Current Phase:** Alpha with Critical Forensics Issues  
**Target:** Production-Ready, FAANG-Level Excellence

---

## Executive Summary

ConfirmIT is an AI-powered, blockchain-anchored trust verification platform designed to combat the ₦5 billion annual fraud crisis in Nigerian commerce. Our solution enables instant verification of receipts, account numbers, and business legitimacy using multi-agent AI forensics combined with immutable blockchain anchoring on Hedera.

**Critical Issue:** During Hedera Hackathon demo, our receipt verification failed to detect a sophisticated forgery (N1,500 altered to N1.5M on an OPay receipt). This exposes fundamental weaknesses in our forensic detection pipeline that must be resolved before Codematic Hackathon submission.

---

## Problem Space

### Market Reality
- **₦5 billion** lost annually to fraud in Nigeria alone
- Fake receipts used for embezzlement, expense fraud, and money laundering
- Fraudulent bank accounts drain billions from unsuspecting victims
- No reliable, instant verification infrastructure exists
- Manual verification is slow, expensive, and ineffective

### User Pain Points
1. **Consumers:** Get scammed by fake receipts, can't verify business legitimacy before payment
2. **Businesses:** Lose revenue to fraud, can't build trust with customers online
3. **Enterprises:** Employees submit fake expense receipts, no automated verification
4. **Financial Institutions:** Money flows to fraudulent accounts daily

---

## Solution Architecture

### Technology Stack

#### Frontend (React/TypeScript)
```
Framework: React 18 + Vite + TypeScript
State Management: Zustand (lightweight, performant)
UI Components: Radix UI + Tailwind CSS
Real-time: Socket.IO client for WebSocket updates
Cloud Storage: Cloudinary (image uploads with transformations)
Forms: React Hook Form + Zod validation
Authentication: Firebase Auth
```

#### Backend Services

**1. NestJS API Gateway** (`backend/`)
- **Role:** Primary API orchestrator, business logic, Firebase integration
- **Port:** 8080
- **Key Responsibilities:**
  - User authentication & authorization (Firebase Admin SDK)
  - Business registration & verification workflows
  - Account fraud reporting system
  - Hedera blockchain integration (HCS topic submissions, HTS NFT minting)
  - Payment gateway integration (Paystack)
  - WebSocket gateway for real-time progress updates
  - Database operations (Firestore)
  
**2. FastAPI AI Service** (`ai-service/`)
- **Role:** Multi-agent AI orchestration for receipt forensics
- **Port:** 8000
- **Key Responsibilities:**
  - Receipt image analysis via multi-agent system
  - OCR extraction using Google Generative AI (Gemini)
  - Forensic manipulation detection
  - Metadata analysis (EXIF data extraction)
  - Business reputation scoring
  - Final reasoning & recommendation synthesis

**3. Firebase Backend**
- **Firestore:** Primary database for receipts, businesses, users, fraud reports
- **Firebase Storage:** Alternative to Cloudinary (currently unused)
- **Firebase Auth:** Email/password + Google OAuth
- **Security Rules:** RLS-style policies on Firestore collections

**4. Hedera Blockchain**
- **Consensus Service (HCS):** Immutable audit trail for verifications
- **Token Service (HTS):** "TrustID" NFTs for verified businesses
- **Network:** Testnet (transitioning to Mainnet)

**5. Cloudinary**
- **Role:** Receipt image storage with automatic transformations
- **Security:** Encrypted uploads, signed URLs
- **CDN:** Global image delivery

---

## Core Feature: Receipt Verification (QuickScan)

### User Flow
1. User uploads receipt image (drag-drop, camera, or file picker)
2. Image uploads to Cloudinary (encrypted, returns `publicId` and `secureUrl`)
3. Backend creates receipt record in Firestore with `status: 'processing'`
4. AI Service receives analysis request via HTTP POST
5. Multi-agent AI pipeline processes receipt:
   - **Vision Agent:** Extracts text, merchants, amounts via Gemini Vision
   - **Forensic Agent:** Detects image manipulation (CURRENTLY WEAK)
   - **Metadata Agent:** Analyzes EXIF data for tampering signs
   - **Reputation Agent:** Cross-checks merchant against business database
   - **Reasoning Agent:** Synthesizes findings into trust score (0-100) and verdict
6. Real-time progress updates via WebSocket (Socket.IO)
7. Results displayed with trust score gauge, verdict badge, forensic details
8. Optional: Anchor result to Hedera HCS for immutable proof

### Technical Implementation

**Frontend:** `src/pages/QuickScan.tsx`
```typescript
// State management with Zustand
useReceiptStore() -> tracks currentReceipt, analysisProgress, results

// File upload hook
useReceiptUpload() -> uploads to Cloudinary via backend proxy

// WebSocket connection
useWebSocket() -> subscribes to receipt ID, receives progress/complete/error events

// Result display
<ResultsDisplay /> -> TrustScoreGauge, ForensicDetails, HederaBadge, FraudReportModal
```

**Backend API:** `backend/src/modules/receipts/receipts.controller.ts`
```typescript
POST /api/receipts/scan
- Validates file (max 20MB, image types only)
- Uploads to Cloudinary (folder: 'receipts/')
- Creates Firestore record
- Forwards to AI Service: POST http://ai-service:8000/api/receipts/analyze
- Returns { receiptId, message, cloudinaryUrl }

WebSocket: /ws/receipts
- Client subscribes to receiptId
- AI Service sends progress updates: upload_complete, ocr_started, forensics_running, analysis_complete
- Frontend receives real-time updates
```

**AI Service:** `ai-service/app/routers/receipts.py`
```python
POST /api/receipts/analyze
- Receives { imageUrl, publicId, receiptId, anchorOnHedera? }
- Orchestrates multi-agent analysis via orchestrator.py
- Sends WebSocket progress updates to backend
- Returns comprehensive analysis result

Multi-Agent Pipeline:
1. vision_agent.py -> Gemini 2.0 Flash (multimodal) -> OCR + entity extraction
2. forensic_agent.py -> Gemini 2.0 Flash -> Manipulation detection (TEXT-BASED ONLY - CRITICAL FLAW)
3. metadata_agent.py -> PIL + piexif -> EXIF analysis
4. reputation_agent.py -> Firestore lookup -> Merchant verification
5. reasoning_agent.py -> Gemini 2.0 Flash -> Final synthesis
```

---

## Critical Architecture Flaw: Forensic Detection

### The Demo Failure

**What Happened:**
- Forged OPay receipt: Changed N1,500 → N1.5M using image editing
- ConfirmIT verdict: **"Authentic"** with 85/100 trust score
- **Why it failed:** Forensic agent only analyzes text prompts, not pixel-level forgery

### Current Forensic Agent Limitations

**File:** `ai-service/app/agents/forensic_agent.py`

```python
async def analyze(self, image_url: str, receipt_data: dict) -> dict:
    prompt = f"""
    Analyze this receipt image for signs of manipulation:
    - Merchant: {receipt_data.get('merchant', 'Unknown')}
    - Amount: {receipt_data.get('amount', 'Unknown')}
    - Date: {receipt_data.get('date', 'Unknown')}
    
    Check for:
    1. Font inconsistencies
    2. Alignment issues
    3. Color variations
    4. Digital artifacts
    """
    
    # PROBLEM: Only sends text prompt to Gemini, not actual forensic analysis
    result = await gemini_vision_model.generate_content([prompt, image_url])
```

**What's Missing:**
1. **Pixel-level analysis:** No detection of cloned regions, histogram anomalies, JPEG compression artifacts
2. **Error Level Analysis (ELA):** Not implemented (detects areas re-saved at different quality levels)
3. **Noise Pattern Analysis:** Missing (genuine receipts have uniform noise, forgeries have inconsistent noise)
4. **Template Matching:** No database of known receipt formats (OPay, Flutterwave, Paystack templates)
5. **Font Forensics:** Not checking if fonts match known receipt printers
6. **ML-based Detection:** No dedicated forgery detection model (e.g., ManTraNet, IID-Net)

### Why This is Critical

- **Trust Erosion:** If deployed, users will lose faith when obvious fakes pass verification
- **Liability Risk:** Businesses relying on ConfirmIT for expense verification will suffer losses
- **Competitive Weakness:** Other solutions with better forensics will dominate
- **Hackathon Disqualification:** Judges will test with forged receipts (they always do)

---

## What We've Achieved

### ✅ Working Features

1. **Full-Stack Infrastructure**
   - React frontend deployed on Vercel
   - NestJS backend ready for deployment (currently local)
   - FastAPI AI service ready for deployment (currently local)
   - Firebase Firestore operational
   - Cloudinary integration functional

2. **Authentication System**
   - Email/password sign-up and login
   - Google OAuth integration
   - Protected routes with Firebase Auth guards

3. **Business Directory**
   - Multi-tier registration (Basic, Verified, Premium)
   - Document upload for verification (CAC, ID, bank statements)
   - TrustID NFT minting on Hedera (HTS integration complete)
   - Public business profiles with trust scores
   - Business dashboard with analytics (API usage, verification stats)

4. **Account Check Feature**
   - Nigerian bank account number verification (10-digit format)
   - Fraud report submission and viewing
   - Risk level scoring based on fraud reports
   - Integration with business directory (shows if account belongs to verified business)

5. **Hedera Integration**
   - HCS topic creation and message submission (immutable audit trail)
   - HTS NFT creation for verified businesses (TrustID tokens)
   - Explorer URL generation for blockchain transparency

6. **Payment Gateway**
   - Paystack integration for business registration payments
   - Webhook handling for payment verification
   - Multi-payment option UI (cards, bank transfer, crypto - crypto not yet functional)

7. **Real-time Updates**
   - WebSocket implementation with Socket.IO
   - Progress updates during receipt analysis
   - Auto-reconnection on connection loss

### 🚧 Partially Working / Needs Improvement

1. **Receipt Verification (QuickScan)**
   - ✅ Upload flow works
   - ✅ OCR extraction accurate (Gemini Vision)
   - ✅ Merchant identification functional
   - ✅ Real-time progress updates operational
   - ❌ **Forensic detection critically flawed** (main blocker)
   - ⚠️ Metadata analysis basic (EXIF only, not comprehensive)
   - ⚠️ Trust score algorithm needs calibration

2. **AI Multi-Agent System**
   - ✅ Agent orchestration works
   - ✅ Vision agent (OCR) excellent
   - ❌ Forensic agent ineffective (see above)
   - ⚠️ Reputation agent limited by small business database
   - ⚠️ Reasoning agent synthesis generic (needs domain expertise)

3. **Business Verification Workflow**
   - ✅ Registration flow complete
   - ⚠️ Admin review dashboard basic (needs better UX)
   - ❌ Document verification manual (not AI-assisted)
   - ❌ KYC/AML compliance checks missing

---

## Big Picture Vision

### Short-Term (Codematic Hackathon - Next 2 Weeks)

**Must-Have:**
1. **Fix Forensic Detection**
   - Implement pixel-level forgery detection
   - Add Error Level Analysis (ELA)
   - Integrate template matching for common receipt formats
   - Achieve 95%+ accuracy on test dataset with forged receipts

2. **Receipt Format Knowledge Base**
   - Build database of legitimate receipt templates (OPay, Paystack, Flutterwave, etc.)
   - Extract font fingerprints from known printers
   - Store typical value ranges (font sizes, spacing, layout coordinates)

3. **Improved Trust Score Algorithm**
   - Weight forensic findings more heavily
   - Penalize metadata inconsistencies (e.g., "created" date after "modified" date)
   - Cross-reference merchant reputation
   - Flag suspicious patterns (unusual amounts, irregular timestamps)

4. **Deploy Backend Services**
   - Deploy FastAPI AI service to Google Cloud Run (with GPU/TPU for faster inference)
   - Deploy NestJS backend to Google Cloud Run or Render
   - Configure production environment variables
   - Set up Cloud Logging and monitoring

5. **Demo-Ready Experience**
   - Polished UI animations (Framer Motion)
   - Error handling for all edge cases
   - Loading skeletons during analysis
   - Shareable verification reports (PDF export)
   - Mobile-optimized interface

**Nice-to-Have:**
- AI-powered business document verification (automate admin review)
- SMS notifications for account fraud alerts
- Bulk receipt upload API for enterprises
- ConfirmIT SDK for developer integration

### Medium-Term (Post-Hackathon - 3-6 Months)

1. **Scale AI Infrastructure**
   - Fine-tune custom forgery detection model on African receipt dataset
   - Implement caching for repeat verifications (Redis)
   - GPU-accelerated inference (NVIDIA T4 or A100)
   - A/B testing different AI models (Gemini vs GPT-4 Vision vs Claude 3.5 Sonnet)

2. **Expand Coverage**
   - Support more African countries (Kenya, Ghana, South Africa)
   - Add invoice verification (not just receipts)
   - Vehicle documentation verification (for auto sales fraud)
   - Property deed verification (for real estate fraud)

3. **Enterprise Features**
   - Dedicated tenant instances
   - Custom approval workflows
   - Advanced analytics dashboard (fraud trends, hotspot analysis)
   - Compliance reporting (audit trails, regulatory exports)

4. **Monetization**
   - Freemium tier (5 scans/month free)
   - Pro tier (100 scans/month, ₦25k/month)
   - Enterprise tier (unlimited, custom pricing)
   - API access (pay-per-scan model)

5. **Partnerships**
   - Integrate with accounting software (QuickBooks, Zoho Books)
   - Partner with banks for account verification API
   - Collaborate with EFCC, ICPC for fraud intelligence sharing
   - Integrate with e-commerce platforms (Jumia, Konga)

### Long-Term (1-2 Years)

1. **Pan-African Trust Infrastructure**
   - Become the default verification layer for African commerce
   - Process 10M+ verifications per month
   - Database of 1M+ verified businesses
   - Real-time fraud intelligence network

2. **Decentralized Trust Network**
   - Community-driven fraud reporting with token incentives
   - Federated learning for privacy-preserving fraud detection
   - Open API for third-party verification integrations

3. **Regulatory Compliance Automation**
   - KYC/AML compliance as a service
   - Automated regulatory reporting
   - Integration with government verification systems (BVN, NIN, CAC)

---

## Technical Debt & Optimization Opportunities

### High Priority

1. **Forensic Detection Overhaul** (CRITICAL)
   - Research: Review academic papers on image forensics (ManTraNet, IID-Net, TruFor)
   - Dataset: Collect 10k+ legitimate receipts + create synthetic forgeries
   - Model: Fine-tune or build custom forgery detection model
   - Validation: Achieve 95%+ precision on held-out test set

2. **Backend Deployment**
   - Deploy AI service to Google Cloud Run with GPU
   - Deploy NestJS backend to Cloud Run or Render
   - Set up Cloud SQL (or keep Firestore, benchmark first)
   - Configure Cloud Storage for receipt images (vs Cloudinary cost analysis)

3. **Security Hardening**
   - Implement rate limiting (Redis-based)
   - Add CAPTCHA to prevent bot abuse
   - Encrypt sensitive data at rest (business documents)
   - Set up WAF (Cloudflare) for DDoS protection

4. **Error Handling & Resilience**
   - Retry logic for AI API failures
   - Circuit breaker pattern for external services
   - Dead letter queue for failed WebSocket messages
   - Graceful degradation when AI service is down

### Medium Priority

5. **Performance Optimization**
   - Image compression before upload (client-side)
   - Lazy loading for business directory
   - Code splitting for faster initial load
   - CDN for static assets (Vercel Edge Network already in use)

6. **Observability**
   - Structured logging (Winston on backend, Pino for AI service)
   - Error tracking (Sentry integration)
   - Performance monitoring (Cloud Trace, Datadog)
   - Cost monitoring (GCP billing alerts)

7. **Testing Coverage**
   - Unit tests for critical paths (forensic agent, trust score calculation)
   - Integration tests for API endpoints
   - E2E tests for user flows (Playwright)
   - Load testing (k6 or Artillery)

### Low Priority

8. **Code Quality**
   - Refactor large components (QuickScan.tsx is 380+ lines)
   - Extract business logic from controllers (NestJS services)
   - Type safety improvements (stricter TypeScript config)
   - ESLint + Prettier consistency

9. **Developer Experience**
   - Docker Compose for local development
   - Seed scripts for test data
   - API documentation (Swagger UI already set up)
   - Contribution guidelines

---

## Key Questions for Consultant Review

### Architecture & Design

1. **Monorepo vs Polyrepo:** Should we consolidate `frontend/`, `backend/`, `ai-service/` into a monorepo (Turborepo/Nx)?
2. **Database Choice:** Firestore vs Cloud SQL (PostgreSQL) for better relational queries and RLS?
3. **Message Queue:** Should we decouple AI processing with a queue (Cloud Tasks, Bull) instead of synchronous HTTP?
4. **Caching Strategy:** Where should we add caching (Redis)? Receipt analysis results? Business profiles?

### AI & Forensics

5. **Forensic Detection Approach:** 
   - Should we build a custom CNN model for forgery detection?
   - Or integrate existing libraries (python-forensics, foto-forensics API)?
   - Or fine-tune a vision model (CLIP, DINOv2) on forged receipts?

6. **Model Selection:**
   - Is Gemini 2.0 Flash optimal for OCR + forensics, or should we use specialized models?
   - Should we try GPT-4 Vision, Claude 3.5 Sonnet, or Llama 3.2 Vision for comparison?

7. **Prompt Engineering:**
   - Are our agent prompts effective? (see `ai-service/app/agents/*`)
   - Should we use structured output (JSON schema enforcement) instead of free-form text?

8. **Agent Orchestration:**
   - Is our sequential agent flow optimal? (Vision → Forensic → Metadata → Reputation → Reasoning)
   - Should we parallelize independent agents (Forensic + Metadata + Reputation simultaneously)?

### Infrastructure & Scaling

9. **Deployment Strategy:**
   - Google Cloud Run (serverless) vs GKE (Kubernetes) for AI service?
   - How to handle GPU inference at scale? (Cloud Run GPU support is limited)

10. **Cost Optimization:**
    - Gemini API costs will scale with usage. Should we self-host Llama 3.2 Vision on our own GPUs?
    - Cloudinary vs Cloud Storage cost comparison at 100k+ images/month?

11. **High Availability:**
    - How to handle AI service downtime gracefully?
    - Should we implement multi-region deployment for lower latency (Nigeria, Kenya, South Africa)?

### Security & Compliance

12. **Data Privacy:**
    - Should receipt images be auto-deleted after N days (GDPR-style right to erasure)?
    - How to handle PII in receipts (customer names, account numbers)?

13. **Fraud Intelligence:**
    - Should we build a shared fraud database (anonymized) across all users?
    - How to prevent attackers from gaming the trust score system?

14. **Audit Trail:**
    - Is Hedera HCS the right choice for immutable logs, or should we use Google Chronicle, AWS Ledger?

### Product & UX

15. **Trust Score Transparency:**
    - Should we show detailed scoring breakdown to users (OCR confidence, forensic flags, reputation)?
    - Or keep it simple with just a score + verdict?

16. **False Positive Handling:**
    - What if we flag a legitimate receipt as fake? How should users dispute/appeal?

17. **Business Model:**
    - Is freemium + subscription optimal, or should we do pay-per-scan?
    - What's the right pricing for African SMEs (most can't afford $50/month)?

---

## Immediate Action Items (This Week)

### For Engineering Team

1. **Research Forensic Detection Solutions** (2-3 days)
   - Review papers: ManTraNet, IID-Net, TruFor, IFosn
   - Test open-source libraries: foto-forensics, python-forensics
   - Evaluate cloud APIs: Google Cloud Vision (image properties), AWS Rekognition (detect moderation)
   - Decision: Build vs Buy vs Integrate

2. **Build Receipt Forgery Test Dataset** (1 day)
   - Collect 100 legitimate receipts (OPay, Paystack, Flutterwave, POS terminals)
   - Create 100 forgeries (Photoshop: alter amounts, dates, merchants)
   - Label ground truth for validation

3. **Implement Baseline Forensic Improvements** (2 days)
   - Add Error Level Analysis (ELA) using PIL
   - Implement noise pattern analysis
   - Build receipt template matcher (compare layout to known formats)
   - Re-test with forged OPay receipt from demo

4. **Deploy Backend to Staging** (1 day)
   - Deploy AI service to Google Cloud Run (us-central1)
   - Deploy NestJS backend to Render (free tier for now)
   - Test end-to-end flow on staging

5. **Optimize Trust Score Algorithm** (1 day)
   - Increase forensic weight in final score (currently too low)
   - Add penalty for metadata inconsistencies
   - Calibrate thresholds for "authentic" vs "suspicious" vs "fraudulent" verdicts

### For Consultant (Claude)

1. **Architecture Review**
   - Evaluate monorepo feasibility
   - Recommend database strategy (Firestore vs SQL)
   - Suggest caching and queueing approach

2. **AI Strategy**
   - Propose forensic detection solution (build vs integrate)
   - Recommend model selection for OCR + forensics
   - Review and optimize agent prompts

3. **Scaling Plan**
   - Design infrastructure for 10k+ daily scans
   - Cost projections (GCP + Gemini API + Cloudinary)
   - High availability strategy

4. **Security Audit**
   - Identify vulnerabilities in current architecture
   - Recommend compliance requirements (GDPR, NDPR)
   - Suggest fraud prevention measures

---

## Conclusion

ConfirmIT has a solid foundation: full-stack architecture, real-time WebSocket updates, Hedera blockchain integration, and a multi-agent AI system. However, the **forensic detection flaw is existential** — without fixing it, the product is fundamentally broken.

**Our hypothesis:** With proper pixel-level forensic analysis + template matching + ML-based forgery detection, we can achieve 95%+ accuracy and build a world-class verification platform.

**The stakes:** If we nail this for Codematic Hackathon, ConfirmIT could become the trust infrastructure for African commerce — a billion-dollar opportunity.

**We need your expertise** to:
1. Validate our architecture decisions
2. Recommend the fastest path to robust forensic detection
3. Design a scalable, cost-effective infrastructure
4. Identify blind spots we haven't considered

Jazakumullahu Khayran. Let's build something transformative.

---

**Prepared by:** ConfirmIT Engineering Team  
**Date:** 2025-11-19  
**For Review By:** Claude (Principal Engineer & Consultant)  
**Status:** Draft v1.0 — Awaiting Feedback
