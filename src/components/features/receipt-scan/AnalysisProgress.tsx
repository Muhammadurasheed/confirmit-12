import React from 'react';
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
  receiptId?: string;
}

interface AgentStatus {
  name: string;
  status: 'pending' | 'running' | 'completed';
  message?: string;
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

export const AnalysisProgress = ({ progress, status, message, currentAgent, agentDetails, receiptId }: AnalysisProgressProps) => {
  const [agentStatuses, setAgentStatuses] = React.useState<Record<string, AgentStatus>>({
    vision: { name: 'Vision Agent', status: 'pending' },
    forensic: { name: 'Forensic Agent', status: 'pending' },
    metadata: { name: 'Metadata Agent', status: 'pending' },
    reputation: { name: 'Reputation Agent', status: 'pending' },
  });

  // Update agent statuses based on current agent
  React.useEffect(() => {
    if (currentAgent) {
      setAgentStatuses(prev => {
        const updated = { ...prev };
        // Mark all agents before current as completed
        const agentOrder = ['vision', 'forensic', 'metadata', 'reputation'];
        const currentIndex = agentOrder.indexOf(currentAgent);
        
        agentOrder.forEach((agent, index) => {
          if (index < currentIndex) {
            updated[agent] = { ...updated[agent], status: 'completed' };
          } else if (agent === currentAgent) {
            updated[agent] = { ...updated[agent], status: 'running', message: message };
          }
        });
        
        return updated;
      });
    }
  }, [currentAgent, message]);

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
              className="pt-4 border-t space-y-3"
            >
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>
                  {isForensicActive 
                    ? '🔬 Multi-layer forensic detection active...' 
                    : 'Multi-agent AI system analyzing your receipt...'}
                </span>
              </div>
              
              {/* Real-time Agent Activity Log */}
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground">🤖 AI Agents Activity:</p>
                <div className="space-y-1.5">
                  {Object.entries(agentStatuses).map(([key, agent]) => (
                    <motion.div
                      key={key}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={`flex items-center gap-2 text-xs p-2 rounded ${
                        agent.status === 'running' ? 'bg-primary/10 border border-primary/20' : 
                        agent.status === 'completed' ? 'bg-green-500/10' : 'bg-muted/50'
                      }`}
                    >
                      {agent.status === 'running' && (
                        <Loader2 className="h-3 w-3 animate-spin text-primary" />
                      )}
                      {agent.status === 'completed' && (
                        <Check className="h-3 w-3 text-green-500" />
                      )}
                      {agent.status === 'pending' && (
                        <div className="h-3 w-3 rounded-full border-2 border-muted-foreground/30" />
                      )}
                      <span className={`font-medium ${
                        agent.status === 'running' ? 'text-primary' :
                        agent.status === 'completed' ? 'text-green-600 dark:text-green-400' :
                        'text-muted-foreground'
                      }`}>
                        {agent.name}
                      </span>
                      {agent.message && agent.status === 'running' && (
                        <span className="text-muted-foreground ml-auto truncate max-w-[200px]">
                          {agent.message}
                        </span>
                      )}
                    </motion.div>
                  ))}
                </div>
              </div>
              
              {isForensicActive && (
                <div className="mt-3 p-3 bg-purple-500/10 border border-purple-500/20 rounded text-xs space-y-2">
                  <p className="font-semibold text-purple-700 dark:text-purple-300 flex items-center gap-2">
                    <span>🔬</span> Deep Forensic Analysis Active
                  </p>
                  <ul className="space-y-1 text-muted-foreground pl-4">
                    <li>✓ Pixel-level manipulation detection</li>
                    <li>✓ Error Level Analysis (ELA) heatmap</li>
                    <li>✓ Clone region detection</li>
                    <li>✓ Template matching verification</li>
                    <li>✓ Metadata integrity checks</li>
                  </ul>
                  <p className="text-xs text-purple-600 dark:text-purple-400 italic mt-2">
                    Analyzing over 50 forensic markers...
                  </p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </Card>
  );
};
