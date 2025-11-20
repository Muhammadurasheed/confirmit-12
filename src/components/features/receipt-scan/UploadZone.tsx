import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Camera, FileImage, X, Shield, CheckCircle2, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'sonner';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  onScanClick?: (file: File) => void;
  disabled?: boolean;
  isScanning?: boolean;
}

export const UploadZone = ({ onFileSelected, onScanClick, disabled, isScanning }: UploadZoneProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [imageQuality, setImageQuality] = useState<'good' | 'low' | null>(null);

  const validateImageQuality = (file: File): Promise<'good' | 'low'> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const minDimension = Math.min(img.width, img.height);
        resolve(minDimension >= 800 ? 'good' : 'low');
        URL.revokeObjectURL(img.src);
      };
      img.src = URL.createObjectURL(file);
    });
  };

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File size must be less than 10MB', {
          description: 'Please compress your image or choose a smaller file.'
        });
        return;
      }

      // Validate file type
      if (!file.type.startsWith('image/')) {
        toast.error('Please upload an image file', {
          description: 'Supported formats: JPG, PNG, JPEG, WebP'
        });
        return;
      }

      setSelectedFile(file);

      // Check image quality
      const quality = await validateImageQuality(file);
      setImageQuality(quality);
      
      if (quality === 'low') {
        toast.warning('Image resolution is low', {
          description: 'For best results, use a higher quality image (minimum 800px).'
        });
      }

      // Simulate upload progress
      setUploadProgress(0);
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return 95;
          }
          return prev + 5;
        });
      }, 50);

      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result as string);
        setUploadProgress(100);
        setTimeout(() => setUploadProgress(0), 500);
      };
      reader.readAsDataURL(file);

      // Don't trigger scan automatically - wait for user to click "Scan Receipt" button
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp'],
    },
    multiple: false,
    disabled,
  });

  const handleCameraCapture = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.capture = 'environment';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        onDrop([file]);
      }
    };
    input.click();
  }, [onDrop]);

  const clearFile = useCallback(() => {
    setPreview(null);
    setSelectedFile(null);
  }, []);

  if (preview) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="space-y-4"
      >
        <Card className="relative overflow-hidden">
          {uploadProgress > 0 && uploadProgress < 100 && (
            <div className="absolute top-0 left-0 right-0 z-10">
              <Progress value={uploadProgress} className="h-1 rounded-none" />
            </div>
          )}
          
          <img
            src={preview}
            alt="Receipt preview"
            className="w-full h-auto max-h-96 object-contain"
          />
          
          <Button
            variant="destructive"
            size="icon"
            className="absolute top-4 right-4 shadow-lg"
            onClick={clearFile}
            disabled={disabled}
          >
            <X className="h-4 w-4" />
          </Button>
          
          <div className="p-4 bg-muted/50 border-t space-y-3">
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                {selectedFile?.name} ({(selectedFile?.size! / 1024).toFixed(0)} KB)
              </p>
              {imageQuality && (
                <Badge variant={imageQuality === 'good' ? 'default' : 'secondary'}>
                  {imageQuality === 'good' ? (
                    <>
                      <CheckCircle2 className="h-3 w-3 mr-1" />
                      Good Quality
                    </>
                  ) : (
                    <>
                      <AlertCircle className="h-3 w-3 mr-1" />
                      Low Resolution
                    </>
                  )}
                </Badge>
              )}
            </div>
            
            {/* Scan Receipt Button */}
            <Button 
              className="w-full" 
              size="lg"
              onClick={() => {
                if (selectedFile && onScanClick) {
                  onScanClick(selectedFile);
                }
              }}
              disabled={disabled || isScanning}
            >
              {isScanning ? (
                <>
                  <Shield className="h-4 w-4 mr-2 animate-spin" />
                  Scanning Receipt...
                </>
              ) : (
                <>
                  <Shield className="h-4 w-4 mr-2" />
                  Scan Receipt for Authenticity
                </>
              )}
            </Button>
          </div>
        </Card>
      </motion.div>
    );
  }

  // Sample receipt examples for user guidance
  const sampleReceipts = [
    { id: 1, label: "Clear text", quality: "good" },
    { id: 2, label: "Full receipt", quality: "good" },
    { id: 3, label: "Good lighting", quality: "good" },
  ];

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card
          {...getRootProps()}
          className={`border-2 border-dashed transition-all cursor-pointer relative overflow-hidden
            ${isDragActive ? 'border-primary bg-primary/5 scale-[1.02]' : 'border-border hover:border-primary/50'} 
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <input {...getInputProps()} />
          
          {/* Animated dashed border on drag */}
          <AnimatePresence>
            {isDragActive && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 border-4 border-dashed border-primary rounded-lg pointer-events-none"
                style={{
                  animation: 'dash 20s linear infinite',
                }}
              />
            )}
          </AnimatePresence>

          <div className="p-12 text-center space-y-6">
            <motion.div
              animate={isDragActive ? { scale: 1.1 } : { scale: 1 }}
              className="flex justify-center"
            >
              <div className={`p-4 rounded-full transition-colors ${isDragActive ? 'bg-primary/20' : 'bg-primary/10'}`}>
                <Upload className={`h-12 w-12 transition-colors ${isDragActive ? 'text-primary' : 'text-primary'}`} />
              </div>
            </motion.div>

            <div className="space-y-2">
              <h3 className="text-xl font-semibold">
                {isDragActive ? 'Drop receipt here' : 'Upload Receipt'}
              </h3>
              <p className="text-muted-foreground">
                {isDragActive ? 'Release to upload' : 'Drag and drop or click to browse'}
              </p>
              <p className="text-sm text-muted-foreground">
                Supports: PNG, JPG, JPEG, WebP (Max 10MB)
              </p>
            </div>

            <div className="flex gap-4 justify-center flex-wrap">
              <Button variant="outline" className="gap-2" disabled={disabled}>
                <FileImage className="h-4 w-4" />
                Choose File
              </Button>
              <Button
                variant="outline"
                className="gap-2"
                onClick={(e) => {
                  e.stopPropagation();
                  handleCameraCapture();
                }}
                disabled={disabled}
              >
                <Camera className="h-4 w-4" />
                Take Photo
              </Button>
            </div>

            {/* Privacy note */}
            <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground pt-4">
              <Shield className="h-3 w-3" />
              <span>Your receipts are encrypted and never shared</span>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Sample receipts carousel */}
      <div className="space-y-3">
        <p className="text-sm font-medium text-center">What good uploads look like:</p>
        <div className="flex gap-3 overflow-x-auto pb-2">
          {sampleReceipts.map((sample) => (
            <Card key={sample.id} className="flex-shrink-0 w-32 p-3 cursor-pointer hover:shadow-md transition-shadow">
              <div className="aspect-square bg-muted/50 rounded-lg mb-2 flex items-center justify-center">
                <FileImage className="h-8 w-8 text-muted-foreground" />
              </div>
              <div className="flex items-center justify-between">
                <p className="text-xs font-medium">{sample.label}</p>
                <CheckCircle2 className="h-3 w-3 text-success" />
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};
