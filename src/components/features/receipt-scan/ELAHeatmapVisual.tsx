import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertTriangle, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ELAHeatmapVisualProps {
  heatmap: number[] | number[][];
  dimensions?: { width: number; height: number };
  suspiciousRegions?: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    severity: number;
  }>;
  manipulationDetected?: boolean;
}

export const ELAHeatmapVisual = ({
  heatmap,
  dimensions,
  suspiciousRegions = [],
  manipulationDetected = false,
}: ELAHeatmapVisualProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(1);
  const [showRegions, setShowRegions] = useState(true);

  useEffect(() => {
    if (!canvasRef.current || !heatmap || heatmap.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width: number, height: number, flatData: number[];

    if (Array.isArray(heatmap[0])) {
      const heatmap2D = heatmap as number[][];
      height = heatmap2D.length;
      width = heatmap2D[0]?.length || 0;
      flatData = heatmap2D.flat();
    } else {
      flatData = heatmap as number[];
      if (dimensions) {
        width = dimensions.width;
        height = dimensions.height;
      } else {
        width = Math.sqrt(flatData.length);
        height = width;
      }
    }

    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    const max = Math.max(...flatData);
    const min = Math.min(...flatData);
    const range = max - min || 1;

    for (let i = 0; i < flatData.length; i++) {
      const normalized = (flatData[i] - min) / range;
      const pixelIndex = i * 4;

      if (normalized < 0.25) {
        data[pixelIndex] = 0;
        data[pixelIndex + 1] = Math.floor(255 * (normalized * 4));
        data[pixelIndex + 2] = 255;
      } else if (normalized < 0.5) {
        data[pixelIndex] = 0;
        data[pixelIndex + 1] = 255;
        data[pixelIndex + 2] = Math.floor(255 * (1 - (normalized - 0.25) * 4));
      } else if (normalized < 0.75) {
        data[pixelIndex] = Math.floor(255 * ((normalized - 0.5) * 4));
        data[pixelIndex + 1] = 255;
        data[pixelIndex + 2] = 0;
      } else {
        data[pixelIndex] = 255;
        data[pixelIndex + 1] = Math.floor(255 * (1 - (normalized - 0.75) * 4));
        data[pixelIndex + 2] = 0;
      }
      data[pixelIndex + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);

    if (showRegions && suspiciousRegions.length > 0) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      suspiciousRegions.forEach((region) => {
        ctx.strokeRect(region.x, region.y, region.width, region.height);
        ctx.fillStyle = `rgba(239, 68, 68, ${region.severity / 100})`;
        ctx.fillRect(region.x, region.y, region.width, region.height);
      });
    }
  }, [heatmap, dimensions, showRegions, suspiciousRegions]);

  if (!heatmap || heatmap.length === 0) {
    return (
      <Alert>
        <AlertDescription>No ELA heatmap data available</AlertDescription>
      </Alert>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <span>ELA Heatmap Analysis</span>
              {manipulationDetected && <AlertTriangle className="h-5 w-5 text-destructive" />}
            </CardTitle>
            <CardDescription>Error Level Analysis - Red areas indicate potential manipulation</CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}>
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={() => setZoom(Math.min(3, zoom + 0.25))}>
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={() => setShowRegions(!showRegions)}>
              <Maximize2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-auto max-h-[500px] border rounded-lg">
          <canvas
            ref={canvasRef}
            style={{
              transform: `scale(${zoom})`,
              transformOrigin: 'top left',
              imageRendering: 'pixelated',
            }}
            className="max-w-full"
          />
        </div>
        
        <div className="mt-4 flex items-center justify-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-500 rounded" />
            <span className="text-sm text-muted-foreground">Low Error</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-500 rounded" />
            <span className="text-sm text-muted-foreground">Medium Error</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded" />
            <span className="text-sm text-muted-foreground">High Error (Suspicious)</span>
          </div>
        </div>

        {manipulationDetected && (
          <Alert variant="destructive" className="mt-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              {suspiciousRegions.length} suspicious region{suspiciousRegions.length !== 1 ? 's' : ''} detected
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};
