import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { FileText, Copy, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState } from 'react';
import { toast } from 'sonner';

interface OCRTextDisplayProps {
  ocrText: string;
  confidence?: number;
  merchant?: string;
  amount?: string | number;
  date?: string;
}

export const OCRTextDisplay = ({
  ocrText,
  confidence,
  merchant,
  amount,
  date,
}: OCRTextDisplayProps) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(ocrText);
    setCopied(true);
    toast.success('Text copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 80) return 'bg-green-500/10 text-green-700 border-green-200';
    if (conf >= 60) return 'bg-yellow-500/10 text-yellow-700 border-yellow-200';
    return 'bg-red-500/10 text-red-700 border-red-200';
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Extracted Text (OCR)
            </CardTitle>
            <CardDescription>
              Text extracted from receipt using AI vision
            </CardDescription>
          </div>
          {confidence !== undefined && (
            <Badge className={getConfidenceColor(confidence)}>
              {confidence}% Confidence
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Key Information */}
        {(merchant || amount || date) && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-muted/50 rounded-lg">
            {merchant && (
              <div>
                <div className="text-xs text-muted-foreground mb-1">Merchant</div>
                <div className="font-semibold">{merchant}</div>
              </div>
            )}
            {amount && (
              <div>
                <div className="text-xs text-muted-foreground mb-1">Amount</div>
                <div className="font-semibold text-primary">
                  {typeof amount === 'number' ? `₦${amount.toLocaleString()}` : amount}
                </div>
              </div>
            )}
            {date && (
              <div>
                <div className="text-xs text-muted-foreground mb-1">Date</div>
                <div className="font-semibold">{date}</div>
              </div>
            )}
          </div>
        )}

        {/* Full Text */}
        <div className="relative">
          <ScrollArea className="h-[200px] w-full rounded-md border p-4 bg-muted/20">
            <pre className="text-sm font-mono whitespace-pre-wrap">{ocrText || 'No text extracted'}</pre>
          </ScrollArea>
          <Button
            variant="ghost"
            size="sm"
            className="absolute top-2 right-2"
            onClick={handleCopy}
            disabled={!ocrText}
          >
            {copied ? (
              <CheckCircle2 className="h-4 w-4 text-green-500" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </Button>
        </div>

        <div className="text-xs text-muted-foreground">
          {ocrText ? `${ocrText.length} characters extracted` : 'No OCR data available'}
        </div>
      </CardContent>
    </Card>
  );
};
