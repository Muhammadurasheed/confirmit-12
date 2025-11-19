import { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, XCircle, HelpCircle, Flag, Link as LinkIcon, Eye } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import TrustScoreGauge from '@/components/shared/TrustScoreGauge';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { ReportFraudModal } from './ReportFraudModal';
import { ForensicDetailsModal } from './ForensicDetailsModal';
import { HederaAnchorModal } from './HederaAnchorModal';
import HederaBadge from '@/components/shared/HederaBadge';

interface ResultsDisplayProps {
  receiptId: string;
  trustScore: number;
  verdict: 'authentic' | 'suspicious' | 'fraudulent' | 'unclear';
  issues: Array<{
    type: string;
    severity: 'high' | 'medium' | 'low';
    description: string;
  }>;
  recommendation: string;
  forensicDetails: {
    ocr_confidence: number;
    manipulation_score: number;
    metadata_flags: string[];
    forensic_summary?: string;
    techniques_detected?: string[];
    authenticity_indicators?: string[];
  };
  merchant?: {
    name: string;
    verified: boolean;
    trust_score: number;
  } | null;
  hederaAnchor?: {
    transaction_id: string;
    consensus_timestamp: string;
    explorer_url: string;
  } | null;
}

const verdictConfig = {
  authentic: {
    icon: CheckCircle,
    color: 'text-green-500',
    bgColor: 'bg-green-500/10',
    label: 'Authentic',
  },
  suspicious: {
    icon: AlertTriangle,
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-500/10',
    label: 'Suspicious',
  },
  fraudulent: {
    icon: XCircle,
    color: 'text-red-500',
    bgColor: 'bg-red-500/10',
    label: 'Fraudulent',
  },
  unclear: {
    icon: HelpCircle,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    label: 'Unclear',
  },
};

export const ResultsDisplay = ({
  receiptId,
  trustScore,
  verdict,
  issues = [],
  recommendation,
  forensicDetails,
  merchant,
  hederaAnchor,
}: ResultsDisplayProps) => {
  const [showReportModal, setShowReportModal] = useState(false);
  const [showForensicModal, setShowForensicModal] = useState(false);
  const [showHederaModal, setShowHederaModal] = useState(false);
  const [showReportFraudModal, setShowReportFraudModal] = useState(false);

  // Defensive checks and data validation
  const safeIssues = Array.isArray(issues) ? issues : [];
  const safeForensicDetails = forensicDetails || {
    ocr_confidence: 0,
    manipulation_score: 0,
    metadata_flags: [],
  };
  const safeMetadataFlags = Array.isArray(safeForensicDetails.metadata_flags) 
    ? safeForensicDetails.metadata_flags 
    : [];

  // Normalize verdict to ensure it matches verdictConfig keys
  const normalizedVerdict = verdict?.toLowerCase() as keyof typeof verdictConfig;
  const config = verdictConfig[normalizedVerdict] || verdictConfig['unclear'];
  const VerdictIcon = config.icon;

  const handleAnchorToHedera = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8080'}/api/receipts/${receiptId}/anchor`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to anchor to Hedera');
      }

      const data = await response.json();
      return {
        transactionId: data.hedera_anchor.transaction_id,
        explorerUrl: data.hedera_anchor.explorer_url,
      };
    } catch (error) {
      console.error('Hedera anchoring error:', error);
      throw error;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Trust Score & Verdict */}
      <Card className="p-8">
        <div className="flex flex-col md:flex-row items-center gap-8">
          <TrustScoreGauge score={trustScore} size="lg" />
          
          <div className="flex-1 text-center md:text-left space-y-4">
            <div className="flex items-center gap-3 justify-center md:justify-start">
              <div className={`p-3 rounded-full ${config.bgColor}`}>
                <VerdictIcon className={`h-8 w-8 ${config.color}`} />
              </div>
              <div>
                <Badge variant={verdict === 'authentic' ? 'default' : verdict === 'fraudulent' ? 'destructive' : 'secondary'} className="text-lg px-4 py-1">
                  {config.label}
                </Badge>
              </div>
            </div>
            
            <div className="space-y-2">
              <h2 className="text-2xl font-bold">Verification Results</h2>
              <p className="text-muted-foreground">Receipt ID: {receiptId}</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Recommendation Banner */}
      <Card className={`p-6 border-l-4 ${verdict === 'authentic' ? 'border-l-green-500' : verdict === 'fraudulent' ? 'border-l-red-500' : 'border-l-yellow-500'}`}>
        <h3 className="font-semibold mb-2">Recommendation</h3>
        <p className="text-sm">{recommendation}</p>
      </Card>

      {/* Issues */}
      {safeIssues.length > 0 && (
        <Card className="p-6">
          <h3 className="font-semibold mb-4">Detected Issues ({safeIssues.length})</h3>
          <div className="space-y-3">
            {safeIssues.map((issue, index) => (
              <div key={index} className="flex gap-3 p-3 rounded-lg bg-muted/50">
                <Badge variant={issue.severity === 'high' ? 'destructive' : issue.severity === 'medium' ? 'default' : 'secondary'}>
                  {issue.severity}
                </Badge>
                <div className="flex-1">
                  <p className="font-medium text-sm">{issue.type.replace(/_/g, ' ')}</p>
                  <p className="text-sm text-muted-foreground">{issue.description}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Merchant Info */}
      {merchant && (
        <Card className="p-6">
          <h3 className="font-semibold mb-4">Merchant Information</h3>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">{merchant.name}</p>
              <p className="text-sm text-muted-foreground">Trust Score: {merchant.trust_score}/100</p>
            </div>
            {merchant.verified && (
              <Badge variant="default" className="gap-1">
                <CheckCircle className="h-3 w-3" />
                Verified
              </Badge>
            )}
          </div>
        </Card>
      )}

      {/* Forensic Details Accordion */}
      <Accordion type="single" collapsible>
        <AccordionItem value="forensics">
          <AccordionTrigger className="text-base font-semibold">
            🔬 Forensic Analysis Details
          </AccordionTrigger>
          <AccordionContent>
            <Card className="p-6 space-y-6">
              {/* Summary */}
              {forensicDetails.forensic_summary && (
                <div className="p-4 bg-muted/50 rounded-lg">
                  <p className="text-sm font-medium mb-2">Forensic Summary:</p>
                  <p className="text-sm text-muted-foreground">{forensicDetails.forensic_summary}</p>
                </div>
              )}

              {/* Key Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">OCR Confidence</p>
                  <p className="text-2xl font-bold">{safeForensicDetails.ocr_confidence}%</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Manipulation Score</p>
                  <p className="text-2xl font-bold">{safeForensicDetails.manipulation_score}%</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Metadata Flags</p>
                  <p className="text-2xl font-bold">{safeMetadataFlags.length}</p>
                </div>
              </div>

              {/* Authenticity Indicators */}
              {forensicDetails.authenticity_indicators && forensicDetails.authenticity_indicators.length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-3 text-green-600 dark:text-green-400">
                    ✓ Authenticity Indicators:
                  </p>
                  <ul className="space-y-2">
                    {forensicDetails.authenticity_indicators.map((indicator, index) => (
                      <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                        <span className="text-green-500 mt-0.5">✓</span>
                        <span>{indicator}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Red Flags */}
              {forensicDetails.techniques_detected && forensicDetails.techniques_detected.length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-3 text-red-600 dark:text-red-400">
                    ⚠ Red Flags Detected:
                  </p>
                  <ul className="space-y-2">
                    {forensicDetails.techniques_detected.map((technique, index) => (
                      <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                        <span className="text-red-500 mt-0.5">⚠</span>
                        <span>{technique}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Metadata Findings */}
              {safeMetadataFlags.length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-2">Metadata Findings:</p>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {safeMetadataFlags.map((flag, index) => (
                      <li key={index}>{flag}</li>
                    ))}
                  </ul>
                </div>
              )}
            </Card>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      {/* Quick Actions */}
      <Card className="p-4">
        <CardContent className="pt-6">
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" size="sm" onClick={() => setShowForensicModal(true)} className="flex-1">
              <Eye className="h-4 w-4 mr-2" />
              View Detailed Analysis
            </Button>
            <Button variant="outline" size="sm" onClick={() => setShowHederaModal(true)} className="flex-1">
              <LinkIcon className="h-4 w-4 mr-2" />
              {hederaAnchor ? 'View Anchor' : 'Anchor on Hedera'}
            </Button>
            <Button variant="outline" size="sm" onClick={() => setShowReportFraudModal(true)} className="flex-1">
              <Flag className="h-4 w-4 mr-2" />
              Report Fraud
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Hedera Badge */}
      {hederaAnchor && (
        <HederaBadge
          transactionId={hederaAnchor.transaction_id}
          explorerUrl={hederaAnchor.explorer_url}
          consensusTimestamp={hederaAnchor.consensus_timestamp}
        />
      )}

      {/* Modals */}
      <ReportFraudModal
        open={showReportFraudModal}
        onOpenChange={setShowReportFraudModal}
        receiptId={receiptId}
      />
      <ForensicDetailsModal
        open={showForensicModal}
        onOpenChange={setShowForensicModal}
        forensicDetails={forensicDetails}
      />
      <HederaAnchorModal
        open={showHederaModal}
        onOpenChange={setShowHederaModal}
        receiptId={receiptId}
        onAnchor={handleAnchorToHedera}
        existingAnchor={hederaAnchor || undefined}
      />
    </motion.div>
  );
};
