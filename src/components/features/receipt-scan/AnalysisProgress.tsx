import { motion, AnimatePresence } from 'framer-motion';
import { Check, Loader2, AlertCircle } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { EducationalTips } from './EducationalTips';

interface AnalysisProgressProps {
  progress: number;
  status: string;
  message: string;
  currentAgent?: string;
  agentDetails?: Record<string, any>;
}

const agentConfig = {
  orchestrator: { label: 'Orchestrator', color: 'text-blue-500', bg: 'bg-blue-500/10' },
  vision: { label: 'Vision Agent', color: 'text-green-500', bg: 'bg-green-500/10' },
  forensic: { label: 'Forensic Agent', color: 'text-purple-500', bg: 'bg-purple-500/10' },
  metadata: { label: 'Metadata Agent', color: 'text-orange-500', bg: 'bg-orange-500/10' },
  reputation: { label: 'Reputation Agent', color: 'text-cyan-500', bg: 'bg-cyan-500/10' },
  reasoning: { label: 'Reasoning Agent', color: 'text-pink-500', bg: 'bg-pink-500/10' },
};

const statusConfig = {
  upload_complete: { label: 'Upload Complete', icon: Check, color: 'text-green-500' },
  ocr_started: { label: 'Extracting Text', icon: Loader2, color: 'text-blue-500' },
  forensics_running: { label: 'Forensic Analysis', icon: Loader2, color: 'text-purple-500' },
  analysis_complete: { label: 'Analysis Complete', icon: Check, color: 'text-green-500' },
  hedera_anchoring: { label: 'Blockchain Anchoring', icon: Loader2, color: 'text-orange-500' },
  hedera_anchored: { label: 'Verified on Blockchain', icon: Check, color: 'text-green-500' },
  complete: { label: 'Verification Complete', icon: Check, color: 'text-green-500' },
  failed: { label: 'Verification Failed', icon: AlertCircle, color: 'text-red-500' },
};

export const AnalysisProgress = ({ progress, status, message, currentAgent, agentDetails }: AnalysisProgressProps) => {
  const config = statusConfig[status as keyof typeof statusConfig] || {
    label: 'Processing',
    icon: Loader2,
    color: 'text-blue-500',
  };

  const Icon = config.icon;
  const isLoading = Icon === Loader2;
  
  // Get current agent info
  const agentInfo = currentAgent ? agentConfig[currentAgent as keyof typeof agentConfig] : null;
  
  // Forensic-specific statuses
  const forensicStatuses = [
    'forensics_running', 
    'pixel_analysis', 
    'ela_analysis', 
    'template_matching', 
    'metadata_check'
  ];
  const isForensicActive = forensicStatuses.includes(status);

  return (
    <Card className="p-6 space-y-6 relative overflow-hidden">
      {/* Animated background particles */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {[...Array(10)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute h-1 w-1 rounded-full bg-primary/10"
            initial={{ x: Math.random() * 100 + '%', y: Math.random() * 100 + '%' }}
            animate={{
              x: [Math.random() * 100 + '%', Math.random() * 100 + '%'],
              y: [Math.random() * 100 + '%', Math.random() * 100 + '%'],
            }}
            transition={{
              duration: Math.random() * 5 + 5,
              repeat: Infinity,
              repeatType: "reverse"
            }}
          />
        ))}
      </div>

      <div className="relative z-10 space-y-6">
        {/* Agent Status Badge */}
        {agentInfo && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full ${agentInfo.bg} border border-border/50`}
          >
            <div className={`h-2 w-2 rounded-full ${agentInfo.color} animate-pulse`} />
            <span className={`text-sm font-medium ${agentInfo.color}`}>{agentInfo.label}</span>
          </motion.div>
        )}
        
        <div className="flex items-center gap-4">
          <motion.div
            animate={isLoading ? { scale: [1, 1.1, 1] } : {}}
            transition={{ duration: 1.5, repeat: Infinity }}
            className={`p-3 rounded-full bg-background ${config.color}`}
          >
            <Icon className={`h-6 w-6 ${isLoading ? 'animate-spin' : ''}`} />
          </motion.div>
          <div className="flex-1">
            <h3 className="font-semibold text-lg">{config.label}</h3>
            <p className="text-sm text-muted-foreground">{message}</p>
            
            {/* Agent Details */}
            {agentDetails && Object.keys(agentDetails).length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {Object.entries(agentDetails).map(([key, value]) => (
                  <span key={key} className="text-xs px-2 py-0.5 rounded bg-muted">
                    {key}: <span className="font-medium">{String(value)}</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Progress</span>
            <span className="font-medium">{progress}%</span>
          </div>
          <Progress value={progress} className="h-2" />
          <p className="text-xs text-muted-foreground text-center">
            {isForensicActive ? 'Deep forensic analysis in progress...' : 'This usually takes 5-8 seconds'}
          </p>
        </div>

        <EducationalTips />

        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="pt-4 border-t space-y-2"
            >
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>
                  {isForensicActive 
                    ? '🔬 Multi-layer forensic detection active...' 
                    : 'Multi-agent AI system analyzing your receipt...'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-1">
                  <div className={`h-1.5 w-1.5 rounded-full ${status === 'ocr_started' ? 'bg-blue-500 animate-pulse' : 'bg-blue-500/30'}`} />
                  <span>Vision Agent</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className={`h-1.5 w-1.5 rounded-full ${isForensicActive ? 'bg-purple-500 animate-pulse' : 'bg-purple-500/30'}`} />
                  <span>Forensic Agent</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className={`h-1.5 w-1.5 rounded-full bg-orange-500/30`} />
                  <span>Metadata Agent</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className={`h-1.5 w-1.5 rounded-full bg-green-500/30`} />
                  <span>Reputation Agent</span>
                </div>
              </div>
              {isForensicActive && (
                <div className="mt-3 p-2 bg-purple-500/10 rounded text-xs space-y-1">
                  <p className="font-medium text-purple-700 dark:text-purple-300">
                    🔍 Active Forensic Layers:
                  </p>
                  <ul className="space-y-0.5 text-muted-foreground pl-4">
                    <li>• Pixel-level manipulation detection</li>
                    <li>• Error Level Analysis (ELA)</li>
                    <li>• Clone region detection</li>
                    <li>• Template matching verification</li>
                  </ul>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </Card>
  );
};
