import { Upload, Microscope } from "lucide-react";
import { useCallback, useState } from "react";

interface AnalysisScannerProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  isAnalyzing: boolean;
}

export const AnalysisScanner = ({ onFileSelect, selectedFile, isAnalyzing }: AnalysisScannerProps) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  if (isAnalyzing) {
    return (
      <div className="flex flex-col items-center justify-center space-y-8">
        <div className="relative">
          <div className="w-64 h-64 border-4 border-primary rounded-full flex items-center justify-center glass relative overflow-hidden">
            {selectedFile && (
              <img
                src={URL.createObjectURL(selectedFile)}
                alt="Analysis target"
                className="w-full h-full object-cover rounded-full"
              />
            )}
            <div className="absolute inset-0 border-4 border-primary rounded-full scanner-rings"></div>
            <div className="absolute inset-4 border-2 border-primary/60 rounded-full scanner-rings" style={{animationDelay: '0.5s'}}></div>
            <div className="absolute inset-8 border-2 border-primary/40 rounded-full scanner-rings" style={{animationDelay: '1s'}}></div>
          </div>
        </div>
        <div className="text-center space-y-2">
          <h3 className="text-xl font-semibold text-primary glow-pulse">Analyzing Cellular Structure</h3>
          <p className="text-muted-foreground">Hybrid transformer intelligence processing...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center space-y-6">
      <div 
        className={`
          relative w-64 h-64 border-4 rounded-full flex flex-col items-center justify-center
          glass cursor-pointer transition-all duration-300
          ${isDragOver || selectedFile 
            ? 'border-primary glow-primary scanner-pulse' 
            : 'border-border hover:border-primary/50'
          }
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        {selectedFile ? (
          <div className="relative w-full h-full rounded-full overflow-hidden">
            <img
              src={URL.createObjectURL(selectedFile)}
              alt="Selected histopathology sample"
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-primary/20 flex items-center justify-center">
              <Microscope className="w-12 h-12 text-white drop-shadow-lg" />
            </div>
          </div>
        ) : (
          <div className="text-center space-y-4">
            <div className="relative">
              <Upload className="w-16 h-16 text-primary mx-auto scanner-pulse" />
              <div className="absolute inset-0 animate-ping">
                <Upload className="w-16 h-16 text-primary/30 mx-auto" />
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-lg font-medium">Analysis Chamber</p>
              <p className="text-sm text-muted-foreground px-6">
                Upload histopathology image for AI analysis
              </p>
            </div>
          </div>
        )}
        
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileInputChange}
          className="hidden"
        />
      </div>
      
      <div className="text-center space-y-2">
        <p className="text-sm text-muted-foreground">
          Supported formats: JPG, PNG, TIFF
        </p>
        <p className="text-xs text-muted-foreground/60">
          Maximum file size: 10MB
        </p>
      </div>
    </div>
  );
};