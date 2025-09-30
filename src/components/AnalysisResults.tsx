import { CheckCircle, AlertTriangle, Brain, TrendingUp } from "lucide-react";
import { useEffect, useState } from "react";

export interface AnalysisResult {
  prediction: "cancer" | "no_cancer";
  confidence: {
    cancer: number;
    no_cancer: number;
  };
  processingTime?: number;
}

interface AnalysisResultsProps {
  results: AnalysisResult;
}

export const AnalysisResults = ({ results }: AnalysisResultsProps) => {
  const [animatedCancer, setAnimatedCancer] = useState(0);
  const [animatedNoCancer, setAnimatedNoCancer] = useState(0);

  useEffect(() => {
    const timer1 = setTimeout(() => {
      setAnimatedCancer(results.confidence.cancer);
    }, 500);
    
    const timer2 = setTimeout(() => {
      setAnimatedNoCancer(results.confidence.no_cancer);
    }, 700);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [results]);

  const isPrimaryPositive = results.prediction === "cancer";
  const primaryConfidence = isPrimaryPositive ? results.confidence.cancer : results.confidence.no_cancer;

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Main Result */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center space-x-3">
          {isPrimaryPositive ? (
            <AlertTriangle className="w-8 h-8 text-destructive" />
          ) : (
            <CheckCircle className="w-8 h-8 text-success" />
          )}
          <h2 className={`text-4xl font-bold ${
            isPrimaryPositive ? 'text-destructive' : 'text-success'
          }`}>
            {isPrimaryPositive ? 'Cancer Detected' : 'No Cancer Detected'}
          </h2>
        </div>
        
        <p className="text-muted-foreground text-lg">
          Analysis complete with {(primaryConfidence * 100).toFixed(1)}% confidence
        </p>
        
        {results.processingTime && (
          <p className="text-xs text-muted-foreground/60">
            Processing time: {results.processingTime}ms
          </p>
        )}
      </div>

      {/* Confidence Visualization */}
      <div className="space-y-6 glass rounded-2xl p-8">
        <div className="flex items-center space-x-3 mb-6">
          <Brain className="w-6 h-6 text-primary" />
          <h3 className="text-xl font-semibold">Model Confidence</h3>
        </div>
        
        {/* Cancer Confidence */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-destructive" />
              <span className="font-medium text-destructive">Cancer</span>
            </div>
            <span className="text-sm font-mono">
              {(animatedCancer * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-secondary rounded-full h-3 overflow-hidden">
            <div 
              className="h-full bg-destructive progress-fill rounded-full relative"
              style={{ width: `${animatedCancer * 100}%` }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-destructive to-red-400 rounded-full"></div>
            </div>
          </div>
        </div>

        {/* No Cancer Confidence */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-5 h-5 text-success" />
              <span className="font-medium text-success">No Cancer</span>
            </div>
            <span className="text-sm font-mono">
              {(animatedNoCancer * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-secondary rounded-full h-3 overflow-hidden">
            <div 
              className="h-full bg-success progress-fill rounded-full relative"
              style={{ width: `${animatedNoCancer * 100}%` }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-success to-green-400 rounded-full"></div>
            </div>
          </div>
        </div>

        {/* Analysis Details */}
        <div className="pt-4 border-t border-border/50">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-primary" />
              <span className="text-muted-foreground">Model:</span>
              <span className="font-mono">HybridTransformer-v2</span>
            </div>
            <div className="flex items-center space-x-2">
              <Brain className="w-4 h-4 text-primary" />
              <span className="text-muted-foreground">Architecture:</span>
              <span className="font-mono">Vision + Pathology</span>
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="text-center space-y-2 p-4 bg-muted/20 rounded-lg border border-border/50">
        <p className="text-xs text-muted-foreground font-medium">
          ⚠️ For Research Purposes Only
        </p>
        <p className="text-xs text-muted-foreground">
          This AI analysis is for educational and research purposes. Always consult qualified medical professionals for actual diagnosis and treatment decisions.
        </p>
      </div>
    </div>
  );
};