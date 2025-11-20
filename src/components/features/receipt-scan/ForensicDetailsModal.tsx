import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Shield, Eye, FileWarning, Bot, AlertTriangle } from "lucide-react";
import { ELAHeatmapViewer } from "./ELAHeatmapViewer";
import { Progress } from "@/components/ui/progress";

interface ForensicDetailsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  receiptId: string;
  receiptImageUrl?: string;
  forensicDetails: {
    ocr_confidence: number;
    manipulation_score: number;
    metadata_flags: string[];
    forensic_progress?: Array<{
      stage: string;
      message: string;
      progress: number;
      details?: Record<string, any>;
    }>;
    technical_details?: {
      ela_analysis?: {
        manipulation_detected?: boolean;
        statistics?: {
          mean_error: number;
          max_error: number;
          std_error: number;
          bright_pixel_ratio: number;
        };
        suspicious_regions?: Array<{
          x: number;
          y: number;
          width: number;
          height: number;
          severity: number;
          mean_error: number;
          max_error: number;
        }>;
        heatmap?: number[][];
        image_dimensions?: { width: number; height: number };
        techniques?: string[];
      };
      pixel_results?: any;
      template_results?: any;
      metadata_results?: any;
    };
    agent_logs?: Array<{
      agent: string;
      status: string;
      confidence?: number;
      manipulation_score?: number;
      flags?: number;
    }>;
  };
  ocrText?: string;
}

export const ForensicDetailsModal = ({
  open,
  onOpenChange,
  receiptId,
  receiptImageUrl,
  forensicDetails,
  ocrText,
}: ForensicDetailsModalProps) => {
  const elaAnalysis = forensicDetails.technical_details?.ela_analysis;
  const hasELAData = elaAnalysis && (elaAnalysis.heatmap || elaAnalysis.suspicious_regions);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            <DialogTitle>Forensic Analysis Details</DialogTitle>
          </div>
          <p className="text-sm text-muted-foreground">
            Comprehensive breakdown of AI-powered analysis
          </p>
        </DialogHeader>

        <Tabs defaultValue={hasELAData ? "ela" : "overview"} className="w-full">
          <TabsList className="grid w-full" style={{ gridTemplateColumns: hasELAData ? 'repeat(5, 1fr)' : 'repeat(4, 1fr)' }}>
            {hasELAData && (
              <TabsTrigger value="ela">
                <AlertTriangle className="h-4 w-4 mr-2" />
                ELA Heatmap
              </TabsTrigger>
            )}
            <TabsTrigger value="overview">
              <Shield className="h-4 w-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="ocr">
              <Eye className="h-4 w-4 mr-2" />
              OCR Text
            </TabsTrigger>
            <TabsTrigger value="metadata">
              <FileWarning className="h-4 w-4 mr-2" />
              Metadata
            </TabsTrigger>
            <TabsTrigger value="agents">
              <Bot className="h-4 w-4 mr-2" />
              AI Agents
            </TabsTrigger>
          </TabsList>

          {/* ELA Heatmap Tab */}
          {hasELAData && receiptImageUrl && (
            <TabsContent value="ela" className="space-y-4 mt-6">
              <ELAHeatmapViewer
                imageUrl={receiptImageUrl}
                heatmap={elaAnalysis.heatmap}
                suspiciousRegions={elaAnalysis.suspicious_regions}
                imageDimensions={elaAnalysis.image_dimensions}
                statistics={elaAnalysis.statistics}
              />
              
              {/* ELA Techniques Detected */}
              {elaAnalysis.techniques && elaAnalysis.techniques.length > 0 && (
                <Card>
                  <CardContent className="pt-6">
                    <h4 className="font-semibold mb-3">ELA Findings</h4>
                    <ul className="space-y-2">
                      {elaAnalysis.techniques.map((technique, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{technique}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          )}

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6 mt-6">
            <Card>
              <CardContent className="pt-6">
                <h4 className="font-semibold mb-4">Confidence Scores</h4>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium">OCR Confidence</span>
                      <span className="text-sm font-bold">{forensicDetails.ocr_confidence}%</span>
                    </div>
                    <Progress value={forensicDetails.ocr_confidence} className="h-2" />
                  </div>

                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium">Manipulation Risk</span>
                      <span className="text-sm font-bold">{forensicDetails.manipulation_score}%</span>
                    </div>
                    <Progress 
                      value={forensicDetails.manipulation_score} 
                      className="h-2"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Metadata Flags */}
            {forensicDetails.metadata_flags && forensicDetails.metadata_flags.length > 0 && (
              <Card>
                <CardContent className="pt-6">
                  <h4 className="font-semibold mb-3">Metadata Flags</h4>
                  <div className="space-y-2">
                    {forensicDetails.metadata_flags.map((flag, idx) => (
                      <div key={idx} className="flex items-center gap-2">
                        <Badge variant="outline">{flag}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* OCR Text Tab */}
          <TabsContent value="ocr" className="space-y-6 mt-6">
            <Card>
              <CardContent className="pt-6">
                <h4 className="font-semibold mb-3">
                  <Eye className="h-4 w-4 inline mr-2" />
                  Extracted Text
                </h4>
                <div className="bg-muted p-4 rounded-lg">
                  <pre className="text-sm whitespace-pre-wrap font-mono">
                    {ocrText || 'No OCR text available'}
                  </pre>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Confidence: {forensicDetails.ocr_confidence}%
                </p>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Metadata Tab */}
          <TabsContent value="metadata" className="space-y-6 mt-6">
            <Card>
              <CardContent className="pt-6">
                <h4 className="font-semibold mb-3">
                  <FileWarning className="h-4 w-4 inline mr-2" />
                  Image Metadata
                </h4>
                <div className="space-y-2">
                  <p className="text-sm">Receipt ID: <span className="font-mono">{receiptId}</span></p>
                  {forensicDetails.metadata_flags.length > 0 ? (
                    <div className="space-y-1">
                      {forensicDetails.metadata_flags.map((flag, idx) => (
                        <div key={idx} className="flex items-center gap-2">
                          <Badge variant="outline">{flag}</Badge>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No metadata flags detected</p>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* AI Agents Tab */}
          <TabsContent value="agents" className="space-y-6 mt-6">
            <Card>
              <CardContent className="pt-6">
            {/* Agent Execution Logs */}
            <div className="mb-6">
              <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                <Bot className="h-4 w-4" />
                Agent Execution Summary
              </h3>
              <div className="space-y-3">
                {forensicDetails.agent_logs && forensicDetails.agent_logs.length > 0 ? (
                  forensicDetails.agent_logs.map((log, idx) => (
                    <Card key={idx}>
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Bot className="h-4 w-4 text-primary" />
                            <span className="font-medium capitalize">{log.agent} Agent</span>
                          </div>
                          <Badge variant={log.status === 'success' ? 'default' : 'destructive'}>
                            {log.status}
                          </Badge>
                        </div>
                        <div className="space-y-1 text-sm text-muted-foreground">
                          {log.confidence !== undefined && (
                            <div className="flex justify-between">
                              <span>Confidence:</span>
                              <span className="font-medium">{log.confidence}%</span>
                            </div>
                          )}
                          {log.manipulation_score !== undefined && (
                            <div className="flex justify-between">
                              <span>Manipulation Score:</span>
                              <span className="font-medium">{log.manipulation_score}%</span>
                            </div>
                          )}
                          {log.flags !== undefined && (
                            <div className="flex justify-between">
                              <span>Flags Detected:</span>
                              <span className="font-medium">{log.flags}</span>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  ))
                ) : (
                  <Card>
                    <CardContent className="p-8 text-center text-muted-foreground">
                      <Bot className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p>No agent execution logs available</p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>

            {/* Forensic Progress Steps */}
            {forensicDetails.forensic_progress && forensicDetails.forensic_progress.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <Shield className="h-4 w-4" />
                  Forensic Analysis Steps
                </h3>
                <div className="space-y-2">
                  {forensicDetails.forensic_progress.map((step: any, idx: number) => (
                    <Card key={idx} className="border-l-4 border-l-purple-500">
                      <CardContent className="p-3">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs font-semibold uppercase text-purple-600 dark:text-purple-400">
                                {step.stage?.replace(/_/g, ' ')}
                              </span>
                              <span className="text-xs text-muted-foreground">
                                {step.progress}%
                              </span>
                            </div>
                            <p className="text-sm">{step.message}</p>
                            {step.details && Object.keys(step.details).length > 0 && (
                              <div className="mt-2 flex flex-wrap gap-1">
                                {Object.entries(step.details).map(([key, value]) => (
                                  <span key={key} className="text-xs px-2 py-0.5 rounded bg-muted">
                                    {key}: <span className="font-medium">{String(value)}</span>
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
};
