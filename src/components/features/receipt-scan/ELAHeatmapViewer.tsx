import { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface SuspiciousRegion {
  x: number;
  y: number;
  width: number;
  height: number;
  severity: number;
  mean_error: number;
  max_error: number;
}

interface PixelDiff {
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
}

interface ELAHeatmapViewerProps {
  imageUrl: string;
  heatmap?: number[][];
  suspiciousRegions?: SuspiciousRegion[];
  imageDimensions?: { width: number; height: number };
  statistics?: {
    mean_error: number;
    max_error: number;
    std_error: number;
    bright_pixel_ratio: number;
  };
  pixelDiff?: PixelDiff;
}

export const ELAHeatmapViewer = ({
  imageUrl,
  heatmap = [],
  suspiciousRegions = [],
  imageDimensions,
  statistics,
  pixelDiff
}: ELAHeatmapViewerProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [viewMode, setViewMode] = useState<'heatmap' | 'pixel-diff'>('heatmap');
  const [hoveredRegion, setHoveredRegion] = useState<SuspiciousRegion | null>(null);
  const [hoveredHotspot, setHoveredHotspot] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageUrl;
    
    img.onload = () => {
      imageRef.current = img;
      setIsLoading(false);
      drawVisualization();
    };
    
    img.onerror = () => {
      console.error('Failed to load image for ELA visualization');
      setIsLoading(false);
    };
  }, [imageUrl]);

  useEffect(() => {
    if (!isLoading) {
      drawVisualization();
    }
  }, [viewMode, heatmap, suspiciousRegions, pixelDiff]);

  const drawVisualization = () => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    // Set canvas size to match container (max 600px width)
    const maxWidth = 600;
    const scale = Math.min(maxWidth / img.width, 1);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;

    // Draw original image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Draw visualization based on mode
    if (viewMode === 'heatmap') {
      // Draw heatmap overlay
      if (heatmap.length > 0) {
        drawHeatmapGrid(ctx, canvas.width, canvas.height);
      }
      // Draw suspicious region boxes
      if (suspiciousRegions.length > 0) {
        drawSuspiciousRegions(ctx, canvas.width, canvas.height);
      }
    } else if (viewMode === 'pixel-diff' && pixelDiff) {
      // Draw pixel-level difference map
      drawPixelDiffMap(ctx, canvas.width, canvas.height);
      // Draw hotspot boxes
      if (pixelDiff.hotspots?.length > 0) {
        drawHotspots(ctx, canvas.width, canvas.height);
      }
    }
  };

  const drawHeatmapGrid = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const gridSize = heatmap.length;
    const cellWidth = width / gridSize;
    const cellHeight = height / gridSize;

    heatmap.forEach((row, i) => {
      row.forEach((intensity, j) => {
        // Normalize intensity to 0-1
        const normalized = intensity / 255;
        
        // Only show high-intensity areas (>0.3)
        if (normalized > 0.3) {
          const x = j * cellWidth;
          const y = i * cellHeight;
          
          // Color mapping: yellow (low) -> red (high)
          const red = 255;
          const green = Math.floor(255 * (1 - normalized));
          const alpha = normalized * 0.4; // Semi-transparent
          
          ctx.fillStyle = `rgba(${red}, ${green}, 0, ${alpha})`;
          ctx.fillRect(x, y, cellWidth, cellHeight);
        }
      });
    });
  };

  const drawSuspiciousRegions = (ctx: CanvasRenderingContext2D, canvasWidth: number, canvasHeight: number) => {
    const imgWidth = imageDimensions?.width || canvasWidth;
    const imgHeight = imageDimensions?.height || canvasHeight;
    const scaleX = canvasWidth / imgWidth;
    const scaleY = canvasHeight / imgHeight;

    suspiciousRegions.forEach((region) => {
      const x = region.x * scaleX;
      const y = region.y * scaleY;
      const w = region.width * scaleX;
      const h = region.height * scaleY;
      
      // Color based on severity
      const severity = region.severity / 100;
      const red = 255;
      const green = Math.floor(255 * (1 - severity));
      
      // Draw semi-transparent box
      ctx.fillStyle = `rgba(${red}, ${green}, 0, 0.25)`;
      ctx.fillRect(x, y, w, h);
      
      // Draw border
      ctx.strokeStyle = `rgba(${red}, ${green}, 0, 0.8)`;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      
      // Draw severity badge
      if (region.severity > 60) {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
        ctx.fillRect(x, y, 40, 20);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText(`${region.severity}%`, x + 5, y + 14);
      }
    });
  };

  const drawPixelDiffMap = (ctx: CanvasRenderingContext2D, canvasWidth: number, canvasHeight: number) => {
    if (!pixelDiff?.diff_map || pixelDiff.diff_map.length === 0) return;

    const diffMap = pixelDiff.diff_map;
    const mapHeight = diffMap.length;
    const mapWidth = diffMap[0]?.length || 0;

    if (mapWidth === 0 || mapHeight === 0) return;

    // Create temporary canvas for pixel diff
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = mapWidth;
    tempCanvas.height = mapHeight;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    const imageData = tempCtx.createImageData(mapWidth, mapHeight);
    
    // Render pixel differences with color mapping
    for (let y = 0; y < mapHeight; y++) {
      for (let x = 0; x < mapWidth; x++) {
        const intensity = diffMap[y][x];
        const idx = (y * mapWidth + x) * 4;
        
        // Color mapping: black (no change) -> yellow -> red (high change)
        if (intensity < 10) {
          // Transparent for minimal changes
          imageData.data[idx] = 0;
          imageData.data[idx + 1] = 0;
          imageData.data[idx + 2] = 0;
          imageData.data[idx + 3] = 0;
        } else {
          const normalized = Math.min(intensity / 255, 1);
          imageData.data[idx] = 255; // Red
          imageData.data[idx + 1] = Math.floor(255 * (1 - normalized)); // Green (less for higher change)
          imageData.data[idx + 2] = 0; // Blue
          imageData.data[idx + 3] = Math.floor(normalized * 200 + 55); // Alpha (more opaque for higher change)
        }
      }
    }

    tempCtx.putImageData(imageData, 0, 0);
    
    // Draw scaled pixel diff overlay
    ctx.drawImage(tempCanvas, 0, 0, canvasWidth, canvasHeight);
  };

  const drawHotspots = (ctx: CanvasRenderingContext2D, canvasWidth: number, canvasHeight: number) => {
    if (!pixelDiff?.hotspots) return;

    const imgWidth = pixelDiff.dimensions.width;
    const imgHeight = pixelDiff.dimensions.height;
    const scaleX = canvasWidth / imgWidth;
    const scaleY = canvasHeight / imgHeight;

    pixelDiff.hotspots.forEach((hotspot) => {
      const x = hotspot.x * scaleX;
      const y = hotspot.y * scaleY;
      const w = hotspot.width * scaleX;
      const h = hotspot.height * scaleY;
      
      // Draw pulsing border for hotspots
      ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      
      // Draw changed pixel count badge
      const changedPct = ((hotspot.changed_pixels / (hotspot.width * hotspot.height)) * 100).toFixed(0);
      ctx.fillStyle = 'rgba(255, 0, 0, 0.95)';
      ctx.fillRect(x, y - 24, 60, 20);
      ctx.fillStyle = 'white';
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText(`${changedPct}% chg`, x + 4, y - 10);
    });
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (viewMode === 'heatmap') {
      const imgWidth = imageDimensions?.width || canvas.width;
      const imgHeight = imageDimensions?.height || canvas.height;
      const scaleX = canvas.width / imgWidth;
      const scaleY = canvas.height / imgHeight;

      // Check if hovering over a suspicious region
      const region = suspiciousRegions.find(r => {
        const rx = r.x * scaleX;
        const ry = r.y * scaleY;
        const rw = r.width * scaleX;
        const rh = r.height * scaleY;
        return x >= rx && x <= rx + rw && y >= ry && y <= ry + rh;
      });

      setHoveredRegion(region || null);
      setHoveredHotspot(null);
    } else if (viewMode === 'pixel-diff' && pixelDiff) {
      const imgWidth = pixelDiff.dimensions.width;
      const imgHeight = pixelDiff.dimensions.height;
      const scaleX = canvas.width / imgWidth;
      const scaleY = canvas.height / imgHeight;

      // Check if hovering over a hotspot
      const hotspot = pixelDiff.hotspots?.find(h => {
        const hx = h.x * scaleX;
        const hy = h.y * scaleY;
        const hw = h.width * scaleX;
        const hh = h.height * scaleY;
        return x >= hx && x <= hx + hw && y >= hy && y <= hy + hh;
      });

      setHoveredHotspot(hotspot || null);
      setHoveredRegion(null);
    }
  };

  const handleCanvasMouseLeave = () => {
    setHoveredRegion(null);
    setHoveredHotspot(null);
  };

  const getSeverityColor = (severity: number) => {
    if (severity >= 80) return 'destructive';
    if (severity >= 60) return 'destructive';
    if (severity >= 40) return 'default';
    return 'secondary';
  };

  const getSeverityLabel = (severity: number) => {
    if (severity >= 80) return 'CRITICAL';
    if (severity >= 60) return 'HIGH';
    if (severity >= 40) return 'MEDIUM';
    return 'LOW';
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-destructive" />
          <h3 className="font-semibold">
            {viewMode === 'heatmap' ? 'ELA Forensic Heatmap' : 'Pixel-Level Diff Map'}
          </h3>
        </div>
        <div className="flex gap-2">
          {pixelDiff && (
            <div className="flex gap-1 border rounded-lg p-1">
              <Button
                variant={viewMode === 'heatmap' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('heatmap')}
              >
                Heatmap
              </Button>
              <Button
                variant={viewMode === 'pixel-diff' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('pixel-diff')}
              >
                Pixel Diff
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Statistics */}
      {viewMode === 'heatmap' && statistics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card className="p-3">
            <div className="text-xs text-muted-foreground">Mean Error</div>
            <div className="text-lg font-bold">{statistics.mean_error.toFixed(1)}</div>
          </Card>
          <Card className="p-3">
            <div className="text-xs text-muted-foreground">Max Error</div>
            <div className="text-lg font-bold">{statistics.max_error.toFixed(1)}</div>
          </Card>
          <Card className="p-3">
            <div className="text-xs text-muted-foreground">Std Deviation</div>
            <div className="text-lg font-bold">{statistics.std_error.toFixed(1)}</div>
          </Card>
          <Card className="p-3">
            <div className="text-xs text-muted-foreground">Bright Pixels</div>
            <div className="text-lg font-bold">{(statistics.bright_pixel_ratio * 100).toFixed(1)}%</div>
          </Card>
        </div>
      )}

      {viewMode === 'pixel-diff' && pixelDiff?.statistics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card className="p-3">
            <div className="text-xs text-muted-foreground">Changed Pixels</div>
            <div className="text-lg font-bold">{pixelDiff.statistics.changed_pixels.toLocaleString()}</div>
          </Card>
          <Card className="p-3">
            <div className="text-xs text-muted-foreground">Change %</div>
            <div className="text-lg font-bold text-destructive">{pixelDiff.statistics.change_percentage.toFixed(2)}%</div>
          </Card>
          <Card className="p-3">
            <div className="text-xs text-muted-foreground">Max Diff</div>
            <div className="text-lg font-bold">{pixelDiff.statistics.max_difference.toFixed(1)}</div>
          </Card>
          <Card className="p-3">
            <div className="text-xs text-muted-foreground">Hotspots</div>
            <div className="text-lg font-bold">{pixelDiff.hotspots?.length || 0}</div>
          </Card>
        </div>
      )}

      {/* Canvas */}
      <div className="relative border rounded-lg overflow-hidden bg-muted">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-muted-foreground">Loading image...</div>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            className="w-full h-auto cursor-crosshair"
            onMouseMove={handleCanvasMouseMove}
            onMouseLeave={handleCanvasMouseLeave}
          />
        )}
        
        {/* Hover tooltips */}
        {hoveredRegion && (
          <div className="absolute top-4 right-4 bg-background border shadow-lg rounded-lg p-3 max-w-xs z-10">
            <div className="flex items-center gap-2 mb-2">
              <Badge variant={getSeverityColor(hoveredRegion.severity)}>
                {getSeverityLabel(hoveredRegion.severity)}
              </Badge>
              <span className="text-sm font-semibold">{hoveredRegion.severity}% Suspicious</span>
            </div>
            <div className="text-xs space-y-1 text-muted-foreground">
              <div>Mean Error: {hoveredRegion.mean_error.toFixed(2)}</div>
              <div>Max Error: {hoveredRegion.max_error.toFixed(2)}</div>
              <div>Position: ({hoveredRegion.x}, {hoveredRegion.y})</div>
            </div>
          </div>
        )}

        {hoveredHotspot && (
          <div className="absolute top-4 right-4 bg-background border shadow-lg rounded-lg p-3 max-w-xs z-10">
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="destructive">HOTSPOT</Badge>
              <span className="text-sm font-semibold">
                {((hoveredHotspot.changed_pixels / (hoveredHotspot.width * hoveredHotspot.height)) * 100).toFixed(1)}% Changed
              </span>
            </div>
            <div className="text-xs space-y-1 text-muted-foreground">
              <div>Changed Pixels: {hoveredHotspot.changed_pixels.toLocaleString()}</div>
              <div>Intensity: {hoveredHotspot.intensity.toFixed(1)}</div>
              <div>Region: {hoveredHotspot.width}x{hoveredHotspot.height}px</div>
              <div>Position: ({hoveredHotspot.x}, {hoveredHotspot.y})</div>
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <Card className="p-4">
        <h4 className="text-sm font-semibold mb-3">
          {viewMode === 'heatmap' ? 'Heatmap Legend' : 'Pixel Diff Legend'}
        </h4>
        {viewMode === 'heatmap' ? (
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-3">
              <div className="w-8 h-4 bg-gradient-to-r from-yellow-400 to-red-600 rounded" />
              <span className="text-muted-foreground">Yellow → Red: Low to High manipulation probability</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-4 border-2 border-red-500 rounded" />
              <span className="text-muted-foreground">Red boxes: Suspicious regions detected</span>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              <AlertTriangle className="h-3 w-3 inline mr-1" />
              Bright areas indicate regions with inconsistent JPEG compression levels
            </div>
          </div>
        ) : (
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-3">
              <div className="w-8 h-4 bg-gradient-to-r from-yellow-400 to-red-600 rounded" />
              <span className="text-muted-foreground">Shows exact pixel differences: Yellow (minor) → Red (major changes)</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-4 border-2 border-red-500 rounded border-dashed" />
              <span className="text-muted-foreground">Dashed boxes: Hotspots with concentrated pixel changes</span>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              <AlertTriangle className="h-3 w-3 inline mr-1" />
              Pixel-level comparison reveals even subtle edits invisible to the naked eye
            </div>
          </div>
        )}
      </Card>

      {/* Suspicious regions list */}
      {suspiciousRegions.length > 0 && (
        <Card className="p-4">
          <h4 className="text-sm font-semibold mb-3">
            Detected Suspicious Regions ({suspiciousRegions.length})
          </h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {suspiciousRegions
              .sort((a, b) => b.severity - a.severity)
              .slice(0, 10)
              .map((region, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded hover:bg-muted cursor-pointer transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <Badge variant={getSeverityColor(region.severity)}>
                      {region.severity}%
                    </Badge>
                    <span className="text-sm">
                      Region {idx + 1} at ({region.x}, {region.y})
                    </span>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    Error: {region.mean_error.toFixed(1)}
                  </span>
                </div>
              ))}
          </div>
        </Card>
      )}
    </div>
  );
};
