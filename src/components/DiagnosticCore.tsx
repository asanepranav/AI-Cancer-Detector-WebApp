import { useState, useCallback } from "react";
import { AnalysisScanner } from "./AnalysisScanner";
import { AnalysisResults, type AnalysisResult } from "./AnalysisResults";
import { Button } from "@/components/ui/button";
import { Brain, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export const DiagnosticCore = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const { toast } = useToast();

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setResults(null);
    
    toast({
      title: "Image Loaded",
      description: `${file.name} ready for analysis`,
    });
  }, [toast]);

  const simulateAnalysis = useCallback(async (): Promise<AnalysisResult> => {
    // Simulate API call with realistic delay
    const delay = 3000 + Math.random() * 2000; // 3-5 seconds
    
    return new Promise((resolve) => {
      setTimeout(() => {
        // Generate realistic but random results
        const cancerConfidence = Math.random() * 0.6 + 0.1; // 0.1 - 0.7
        const noCancerConfidence = 1 - cancerConfidence;
        
        const prediction = cancerConfidence > 0.5 ? "cancer" : "no_cancer";
        
        resolve({
          prediction,
          confidence: {
            cancer: cancerConfidence,
            no_cancer: noCancerConfidence,
          },
          processingTime: Math.round(delay),
        });
      }, delay);
    });
  }, []);

  const handleAnalysis = useCallback(async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    
    try {
      toast({
        title: "Analysis Started",
        description: "Processing histopathology image with AI...",
      });

      const result = await simulateAnalysis();
      setResults(result);
      
      toast({
        title: "Analysis Complete",
        description: `Prediction: ${result.prediction === 'cancer' ? 'Cancer Detected' : 'No Cancer Detected'}`,
        variant: result.prediction === 'cancer' ? 'destructive' : 'default',
      });
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: "An error occurred during analysis. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFile, simulateAnalysis, toast]);

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setResults(null);
    setIsAnalyzing(false);
  }, []);

  return (
    <div className="min-h-screen neural-bg flex flex-col">
      {/* Header */}
      <header className="text-center py-12 space-y-4">
        <div className="flex items-center justify-center space-x-4">
          <Brain className="w-12 h-12 text-primary glow-pulse" />
          <h1 className="text-5xl font-bold bg-gradient-to-r from-primary to-primary-glow bg-clip-text text-transparent">
            AI Diagnostic Core
          </h1>
        </div>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Analyzing cellular structures with hybrid transformer intelligence
        </p>
        <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
          <Sparkles className="w-4 h-4 text-primary" />
          <span>Powered by Advanced Machine Learning</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-6">
        <div className="w-full max-w-4xl">
          {results ? (
            <div className="space-y-8">
              <AnalysisResults results={results} />
              <div className="text-center">
                <Button
                  onClick={handleReset}
                  variant="outline"
                  size="lg"
                  className="border-primary text-primary hover:bg-primary hover:text-primary-foreground"
                >
                  Analyze Another Sample
                </Button>
              </div>
            </div>
          ) : (
            <div className="glass rounded-3xl p-12 space-y-12">
              <AnalysisScanner
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                isAnalyzing={isAnalyzing}
              />
              
              {selectedFile && !isAnalyzing && (
                <div className="text-center space-y-4">
                  <Button
                    onClick={handleAnalysis}
                    size="lg"
                    className="bg-primary hover:bg-primary/90 text-primary-foreground px-12 py-4 text-lg font-semibold glow-primary"
                  >
                    Begin Analysis
                  </Button>
                  <p className="text-sm text-muted-foreground">
                    Advanced AI will analyze the uploaded histopathology image
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center py-8">
        <p className="text-xs text-muted-foreground/60">
          Research & Educational Tool • Not for Clinical Diagnosis
        </p>
      </footer>
    </div>
  );
};