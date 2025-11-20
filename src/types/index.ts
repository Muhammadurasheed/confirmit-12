// Core Types for ConfirmIT Platform

export type TrustVerdict = "authentic" | "suspicious" | "fraudulent" | "unclear";
export type RiskLevel = "high" | "medium" | "low";
export type IssueSeverity = "high" | "medium" | "low";
export type AnalysisStatus = "pending" | "processing" | "completed" | "failed";
export type BusinessTier = 1 | 2 | 3;
export type VerificationStatus = "pending" | "under_review" | "approved" | "rejected";

// Receipt Types
export interface Receipt {
  receiptId: string;
  userId?: string;
  storagePath: string;
  uploadTimestamp: Date;
  analysis: AnalysisResult | null;
  hederaAnchor?: HederaAnchor;
  status: AnalysisStatus;
  processingTime?: number;
  currentAgent?: string; // Current agent processing the receipt
  agentDetails?: Record<string, any>; // Agent-specific details (merchant, amount, etc.)
}

export interface AnalysisResult {
  trustScore?: number;
  trust_score?: number; // Backend snake_case
  verdict: TrustVerdict;
  issues: Issue[];
  recommendation: string;
  forensicDetails?: ForensicDetails;
  forensic_details?: ForensicDetails; // Backend snake_case
  merchant?: MerchantInfo;
  ocr_text?: string; // OCR extracted text
  agent_logs?: AgentLog[]; // Agent execution logs
}

export interface Issue {
  type: string;
  severity: IssueSeverity;
  description: string;
}

export interface ForensicDetails {
  ocrConfidence?: number;
  ocr_confidence?: number; // Backend snake_case
  manipulationScore?: number;
  manipulation_score?: number; // Backend snake_case
  metadataFlags?: string[];
  metadata_flags?: string[]; // Backend snake_case
  agentLogs?: AgentLog[];
  agent_logs?: AgentLog[]; // Backend snake_case
  forensic_summary?: string; // Summary of forensic analysis
  techniques_detected?: string[]; // Red flags detected
  authenticity_indicators?: string[]; // Authenticity markers
  forensic_progress?: any[]; // Detailed forensic analysis steps
  technical_details?: any; // ELA analysis, forensic data
  heatmap?: number[][];
  suspicious_regions?: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    severity: number;
    mean_error: number;
    max_error: number;
  }>;
  image_dimensions?: {
    width: number;
    height: number;
  };
  mean_error?: number;
  max_error?: number;
  std_error?: number;
  bright_pixel_ratio?: number;
  pixel_diff?: {
    diff_map: number[][];
    dimensions: { width: number; height: number };
    statistics: {
      changed_pixels: number;
      total_pixels: number;
      change_percentage: number;
      max_difference: number;
      mean_difference: number;
    };
    hotspots: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      intensity: number;
      changed_pixels: number;
    }>;
  };
}

export interface AgentLog {
  agent: string;
  status: string;
  timestamp?: Date;
  details?: string;
  confidence?: number;
  manipulation_score?: number;
  flags?: number;
  accounts_checked?: number;
}

export interface MerchantInfo {
  name: string;
  verified: boolean;
  trustScore?: number;
  trust_score?: number; // Backend snake_case
}

export interface HederaAnchor {
  transactionId: string;
  consensusTimestamp: string;
  explorerUrl: string;
  hash: string;
}

// Account Types
export interface AccountCheck {
  accountId: string;
  accountHash: string;
  bankCode?: string;
  trustScore: number;
  riskLevel: RiskLevel;
  checks: AccountCheckDetails;
}

export interface AccountCheckDetails {
  lastChecked: Date;
  checkCount: number;
  fraudReports: FraudReportSummary;
  verifiedBusinessId?: string;
  flags: Issue[];
}

export interface FraudReportSummary {
  total: number;
  recent30Days: number;
  patterns: string[];
}

// Business Types
export interface Business {
  businessId: string;
  name: string;
  category: string;
  logo?: string;
  website?: string;
  linkedin?: string;
  bio?: string;
  contact: ContactInfo;
  bankAccount: BankAccountInfo;
  verification: VerificationInfo;
  trustScore: number;
  rating: number;
  reviewCount: number;
  profileViews?: number;
  verifications?: number;
  fraudReports?: number;
  createdAt?: string;
  stats: BusinessStats;
  apiKeys?: ApiKey[];
  hedera?: HederaInfo;
}

export interface ContactInfo {
  email: string;
  phone: string;
  address: string;
}

export interface BankAccountInfo {
  numberEncrypted: string;
  bankCode: string;
  accountName: string;
  verified: boolean;
}

export interface VerificationInfo {
  tier: BusinessTier;
  status: VerificationStatus;
  verified: boolean;
  verifiedAt?: string;
  documents: VerificationDocuments;
}

export interface VerificationDocuments {
  cacCertificate?: string;
  governmentId?: string;
  proofOfAddress?: string;
  bankStatement?: string;
}

export interface BusinessStats {
  profileViews: number;
  verifications: number;
  successfulTransactions: number;
}

export interface ApiKey {
  keyId: string;
  keyHash: string;
  environment: "test" | "live";
  createdAt: Date;
}

export interface HederaInfo {
  trustIdNft?: {
    tokenId: string;
    serialNumber: string;
    explorerUrl: string;
  };
  // Support snake_case from Firestore
  trust_id_nft?: {
    token_id: string;
    serial_number: string;
    explorer_url: string;
  };
  walletAddress?: string;
}

// Fraud Report Types
export interface FraudReport {
  reportId: string;
  accountHash: string;
  reporterId: string;
  report: FraudReportDetails;
  evidence: FraudEvidence;
  status: "pending" | "verified" | "disputed" | "resolved";
  votes: FraudReportVotes;
  reportedAt: Date;
}

export interface FraudReportDetails {
  businessName: string;
  description: string;
  amountLost: number;
  currency: string;
}

export interface FraudEvidence {
  receiptId?: string;
  screenshots?: string[];
  additionalInfo?: string;
}

export interface FraudReportVotes {
  helpful: number;
  notHelpful: number;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
}

export interface ApiError {
  code: string;
  message: string;
  details?: any;
}
