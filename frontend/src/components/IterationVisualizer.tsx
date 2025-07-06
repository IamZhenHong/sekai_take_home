import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ChevronDown, ChevronRight, Activity, Target, Brain, MessageSquare, List, Wifi, WifiOff } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ContentData {
  content_id: string;
  title: string;
  intro: string;
  character_list: string;
  initial_record: string;
  generated_tags: string;
}

interface RecommendationData {
  recommendation_list: number[];
  precision: number;
  taste_profile: string;
  iteration: number;
  max_iterations: number;
  prompt_feedback: string;
  recommendation_prompt: string;
  recommendation_time: number;
}

interface EvaluationData {
  precision: number;
  prompt_feedback: string;
  taste_profile: string;
  iteration: number;
  max_iterations: number;
  ground_truth_list: number[];
  tag_overlap_ratio: number;
  tag_overlap_count: number;
  semantic_overlap_ratio: number;
  avg_semantic_similarity: number;
  max_semantic_similarity: number;
  evaluation_time: number;
}

interface OptimiserData {
  optimised_prompt: string;
  iteration: number;
  max_iterations: number;
  prompt_feedback: string;
  taste_profile: string;
  ground_truth_list: number[];
  optimiser_time: number;
}

interface IterationData {
  iteration_number: number;
  recommendation: RecommendationData;
  evaluation: EvaluationData;
  optimiser: OptimiserData;
  timestamp?: string;
}

interface DiffIndicatorProps {
  value: number;
  previousValue?: number;
  className?: string;
}

const DiffIndicator: React.FC<DiffIndicatorProps> = ({ value, previousValue, className }) => {
  if (previousValue === undefined) return <span className={className}>{value.toFixed(2)}</span>;
  
  const diff = value - previousValue;
  const isPositive = diff > 0;
  const isZero = diff === 0;
  
  return (
    <span className={cn("flex items-center gap-1", className)}>
      {value.toFixed(2)}
      {!isZero && (
        <span className={cn(
          "text-xs font-medium",
          isPositive ? "text-green-600" : "text-red-600"
        )}>
          ({isPositive ? "↑" : "↓"} {Math.abs(diff).toFixed(2)})
        </span>
      )}
    </span>
  );
};

interface IterationCardProps {
  data: IterationData;
  previousData?: IterationData;
  isExpanded: boolean;
  onToggleExpand: () => void;
  contentData: Record<string, ContentData>;
}

const IterationCard: React.FC<IterationCardProps> = ({ 
  data, 
  previousData, 
  isExpanded, 
  onToggleExpand,
  contentData
}) => {
  const newRecommendations = previousData 
    ? data.recommendation.recommendation_list.filter(id => !previousData.recommendation.recommendation_list.includes(id))
    : [];

  const tasteProfileChanged = previousData && data.recommendation.taste_profile !== previousData.recommendation.taste_profile;

  return (
    <Card className="mb-4 transition-all duration-200 hover:shadow-md">
      <CardHeader 
        className="cursor-pointer select-none pb-3"
        onClick={onToggleExpand}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            <CardTitle className="text-lg font-mono">
              Iteration {data.iteration_number}
            </CardTitle>
            <Badge variant="outline" className="text-xs">
              {data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : 'Live'}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4 text-blue-500" />
            <DiffIndicator 
              value={data.evaluation.precision} 
              previousValue={previousData?.evaluation.precision}
              className="font-mono text-sm"
            />
          </div>
        </div>
      </CardHeader>

      {isExpanded && (
        <CardContent className="pt-0 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Recommendation Section */}
            <div className="space-y-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <List className="h-4 w-4 text-purple-500" />
                  <h4 className="font-semibold text-sm">Recommendation</h4>
                </div>
                <Badge variant="outline" className="text-xs">
                  {data.recommendation.recommendation_time.toFixed(2)}s
                </Badge>
              </div>
              
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <List className="h-4 w-4 text-purple-500" />
                  <h5 className="font-medium text-xs">Recommendation List</h5>
                </div>
                <div className="bg-gray-50 rounded-lg p-3 font-mono text-sm">
                  <div className="space-y-2">
                    {data.recommendation.recommendation_list.map((id, index) => {
                      const content = contentData[id];
                      const isNew = newRecommendations.includes(id);
                      
                      return (
                        <div key={id} className="border rounded p-2 bg-white">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge 
                              variant={isNew ? "default" : "secondary"}
                              className={cn(
                                "text-xs",
                                isNew && "bg-green-100 text-green-800 border-green-300"
                              )}
                            >
                              {id}
                              {isNew && <span className="ml-1">✨</span>}
                            </Badge>
                          </div>
                          {content ? (
                            <div className="text-xs">
                              <div className="font-semibold text-gray-800">{content.title}</div>
                              <div className="text-gray-600 mt-1">{content.intro}</div>
                              {content.character_list && (
                                <div className="text-gray-500 mt-1">Characters: {content.character_list}</div>
                              )}
                              {content.generated_tags && (
                                <div className="text-gray-500 mt-1">Tags: {content.generated_tags}</div>
                              )}
                            </div>
                          ) : (
                            <div className="text-gray-400 text-xs">Loading content...</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Brain className="h-4 w-4 text-indigo-500" />
                  <h5 className="font-medium text-xs">
                    Taste Profile
                    {tasteProfileChanged && <span className="ml-2 text-xs text-orange-600">● Updated</span>}
                  </h5>
                </div>
                <div className="bg-blue-50 rounded-lg p-3 text-sm leading-relaxed">
                  {data.recommendation.taste_profile}
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <MessageSquare className="h-4 w-4 text-cyan-500" />
                  <h5 className="font-medium text-xs">Recommendation Prompt</h5>
                </div>
                <div className="bg-gray-50 rounded-lg p-3 font-mono text-sm leading-relaxed">
                  "{data.recommendation.recommendation_prompt}"
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="h-4 w-4 text-orange-500" />
                  <h5 className="font-medium text-xs">Prompt Feedback</h5>
                </div>
                <div className="bg-yellow-50 rounded-lg p-3 text-sm leading-relaxed">
                  {data.recommendation.prompt_feedback}
                </div>
              </div>
            </div>

            {/* Evaluation Section */}
            <div className="space-y-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-green-500" />
                  <h4 className="font-semibold text-sm">Evaluation</h4>
                </div>
                <Badge variant="outline" className="text-xs">
                  {data.evaluation.evaluation_time.toFixed(2)}s
                </Badge>
              </div>
              
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Target className="h-4 w-4 text-green-500" />
                  <h5 className="font-medium text-xs">Precision</h5>
                </div>
                <div className="bg-green-50 rounded-lg p-3 font-mono text-sm">
                  {data.evaluation.precision.toFixed(3)}
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Target className="h-4 w-4 text-green-500" />
                  <h5 className="font-medium text-xs">Tag Overlap</h5>
                </div>
                <div className="bg-green-50 rounded-lg p-3 font-mono text-sm">
                  <div className="flex justify-between">
                    <span>Ratio: {data.evaluation.tag_overlap_ratio.toFixed(3)}</span>
                    <span>Count: {data.evaluation.tag_overlap_count}</span>
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Target className="h-4 w-4 text-green-500" />
                  <h5 className="font-medium text-xs">Semantic Overlap</h5>
                </div>
                <div className="bg-green-50 rounded-lg p-3 font-mono text-sm">
                  <div className="space-y-1">
                    <div>Ratio: {data.evaluation.semantic_overlap_ratio.toFixed(3)}</div>
                    <div>Avg Similarity: {data.evaluation.avg_semantic_similarity.toFixed(3)}</div>
                    <div>Max Similarity: {data.evaluation.max_semantic_similarity.toFixed(3)}</div>
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <List className="h-4 w-4 text-green-500" />
                  <h5 className="font-medium text-xs">Ground Truth List</h5>
                </div>
                <div className="bg-gray-50 rounded-lg p-3 font-mono text-sm">
                  <div className="space-y-2">
                    {data.evaluation.ground_truth_list.map((id) => {
                      const content = contentData[id];
                      return (
                        <div key={id} className="border rounded p-2 bg-white">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge variant="outline" className="text-xs">
                              {id}
                            </Badge>
                          </div>
                          {content ? (
                            <div className="text-xs">
                              <div className="font-semibold text-gray-800">{content.title}</div>
                              <div className="text-gray-600 mt-1">{content.intro}</div>
                              {content.character_list && (
                                <div className="text-gray-500 mt-1">Characters: {content.character_list}</div>
                              )}
                              {content.generated_tags && (
                                <div className="text-gray-500 mt-1">Tags: {content.generated_tags}</div>
                              )}
                            </div>
                          ) : (
                            <div className="text-gray-400 text-xs">Loading content...</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="h-4 w-4 text-orange-500" />
                  <h5 className="font-medium text-xs">Prompt Feedback</h5>
                </div>
                <div className="bg-yellow-50 rounded-lg p-3 text-sm leading-relaxed">
                  {data.evaluation.prompt_feedback}
                </div>
              </div>
            </div>

            {/* Optimiser Section */}
            <div className="space-y-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4 text-indigo-500" />
                  <h4 className="font-semibold text-sm">Optimiser</h4>
                </div>
                <Badge variant="outline" className="text-xs">
                  {data.optimiser.optimiser_time.toFixed(2)}s
                </Badge>
              </div>
              
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <MessageSquare className="h-4 w-4 text-cyan-500" />
                  <h5 className="font-medium text-xs">Optimised Prompt</h5>
                </div>
                <div className="bg-gray-50 rounded-lg p-3 font-mono text-sm leading-relaxed">
                  "{data.optimiser.optimised_prompt}"
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <List className="h-4 w-4 text-indigo-500" />
                  <h5 className="font-medium text-xs">Ground Truth List</h5>
                </div>
                <div className="bg-gray-50 rounded-lg p-3 font-mono text-sm">
                  <div className="space-y-2">
                    {data.optimiser.ground_truth_list.map((id) => {
                      const content = contentData[id];
                      return (
                        <div key={id} className="border rounded p-2 bg-white">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge variant="outline" className="text-xs">
                              {id}
                            </Badge>
                          </div>
                          {content ? (
                            <div className="text-xs">
                              <div className="font-semibold text-gray-800">{content.title}</div>
                              <div className="text-gray-600 mt-1">{content.intro}</div>
                              {content.character_list && (
                                <div className="text-gray-500 mt-1">Characters: {content.character_list}</div>
                              )}
                              {content.generated_tags && (
                                <div className="text-gray-500 mt-1">Tags: {content.generated_tags}</div>
                              )}
                            </div>
                          ) : (
                            <div className="text-gray-400 text-xs">Loading content...</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="h-4 w-4 text-orange-500" />
                  <h5 className="font-medium text-xs">Prompt Feedback</h5>
                </div>
                <div className="bg-yellow-50 rounded-lg p-3 text-sm leading-relaxed">
                  {data.optimiser.prompt_feedback}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
};

const IterationVisualizer: React.FC = () => {
  const [iterations, setIterations] = useState<IterationData[]>([]);
  const [expandedCards, setExpandedCards] = useState<Set<number>>(new Set());
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [selectedUserId, setSelectedUserId] = useState<string>("");
  const [maxIterations, setMaxIterations] = useState<number>(3);
  const [isLoadingUsers, setIsLoadingUsers] = useState(false);
  const [contentData, setContentData] = useState<Record<string, ContentData>>({});
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:8080/ws/recommend');
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('WebSocket connected - waiting for manual submission');
          setIsConnected(true);
          setConnectionError(null);
          // Explicitly do NOT send any request - wait for user to click button
        };

        ws.onmessage = (event) => {
          try {
            console.log('Raw message received:', event.data);
            const message = JSON.parse(event.data);
            console.log('Parsed message:', message);
            
            if (message.type === 'setup') {
              console.log('Setup received:', message);
              // Handle setup message if needed
            } else if (message.type === 'progress') {
              console.log('Progress:', message.step, 'iteration:', message.iteration);
              // Handle progress updates if needed
            } else if (message.type === 'iteration') {
              const data: IterationData = message;
              data.timestamp = new Date().toISOString();
              
              setIterations(prev => {
                // Update existing iteration or add new one
                const existingIndex = prev.findIndex(i => i.iteration_number === data.iteration_number);
                if (existingIndex >= 0) {
                  const updated = [...prev];
                  updated[existingIndex] = data;
                  return updated;
                } else {
                  return [...prev, data];
                }
              });
              
              // Auto-expand the latest iteration
              setExpandedCards(prev => new Set([...prev, data.iteration_number]));
              
              // Fetch content data for new recommendations and ground truth lists
              const allContentIds = [
                ...data.recommendation.recommendation_list,
                ...data.evaluation.ground_truth_list,
                ...data.optimiser.ground_truth_list
              ];
              fetchContentData(allContentIds);
            } else if (message.type === 'final') {
              console.log('Final state received:', message);
              setIsStreaming(false);
            } else if (message.type === 'error') {
              console.error('WebSocket error:', message.error);
              setConnectionError(message.error);
            } else {
              // Handle legacy format (direct iteration data)
              const data: IterationData = message;
              data.timestamp = new Date().toISOString();
              
              setIterations(prev => {
                const existingIndex = prev.findIndex(i => i.iteration_number === data.iteration_number);
                if (existingIndex >= 0) {
                  const updated = [...prev];
                  updated[existingIndex] = data;
                  return updated;
                } else {
                  return [...prev, data];
                }
              });
              
              setExpandedCards(prev => new Set([...prev, data.iteration_number]));
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        ws.onclose = (event) => {
          console.log('WebSocket disconnected, code:', event.code, 'reason:', event.reason);
          setIsConnected(false);
          setIsStreaming(false);
          
          // Only reconnect if it wasn't a normal closure
          if (event.code !== 1000) {
            setConnectionError('Connection lost. Attempting to reconnect...');
            setTimeout(() => {
              if (wsRef.current?.readyState === WebSocket.CLOSED) {
                connectWebSocket();
              }
            }, 3000);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          setConnectionError('Failed to connect to WebSocket server');
          setIsConnected(false);
          setIsStreaming(false);
        };

      } catch (error) {
        console.error('Error creating WebSocket connection:', error);
        setConnectionError('Failed to create WebSocket connection');
        setIsConnected(false);
        setIsStreaming(false);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []); // Empty dependency array - only run once on mount

  const toggleExpanded = (iteration: number) => {
    setExpandedCards(prev => {
      const newSet = new Set(prev);
      if (newSet.has(iteration)) {
        newSet.delete(iteration);
      } else {
        newSet.add(iteration);
      }
      return newSet;
    });
  };

  const fetchContentData = async (contentIds: number[]) => {
    if (contentIds.length === 0) return;
    
    // Filter out IDs that are already cached
    const uncachedIds = contentIds.filter(id => !contentData[id]);
    if (uncachedIds.length === 0) return; // All content already cached
    
    setIsLoadingContent(true);
    try {
      // Convert numbers to strings and join with commas
      const contentIdsString = uncachedIds.map(id => id.toString()).join(',');
      const response = await fetch(`/api/content/batch?content_ids=${contentIdsString}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      const contentMap: Record<string, ContentData> = {};
      data.content.forEach((item: ContentData) => {
        contentMap[item.content_id] = item;
      });
      
      setContentData(prev => ({ ...prev, ...contentMap }));
    } catch (error) {
      console.error('Error fetching content:', error);
    } finally {
      setIsLoadingContent(false);
    }
  };

  const startNewRecommendation = () => {
    if (!selectedUserId.trim()) {
      setConnectionError('Please enter a user ID');
      return;
    }
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const request = {
        user_id: selectedUserId.trim(),
        k: 5,
        seed: 42,
        max_iterations: maxIterations
      };
      console.log('Starting new recommendation for user:', selectedUserId, 'with', maxIterations, 'iterations');
      wsRef.current.send(JSON.stringify(request));
      setIterations([]);
      setExpandedCards(new Set());
      setContentData({}); // Clear content data
      setIsStreaming(true);
      setConnectionError(null);
    } else {
      setConnectionError('WebSocket not connected. Please wait for connection.');
    }
  };

  const currentIteration = iterations.length > 0 ? iterations[iterations.length - 1] : null;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        {/* Sidebar */}
        <div className="w-80 bg-white border-r border-gray-200 p-6 sticky top-0 h-screen">
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-bold mb-4">Iteration Visualizer</h2>
              
              {/* Connection Status */}
              <div className="flex items-center gap-2 text-sm mb-2">
                {isConnected ? (
                  <>
                    <Wifi className="h-4 w-4 text-green-500" />
                    <span className="text-green-600">Connected</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="h-4 w-4 text-red-500" />
                    <span className="text-red-600">Disconnected</span>
                  </>
                )}
              </div>
              
              {isStreaming && (
                <div className="flex items-center gap-2 text-sm text-blue-600">
                  <div className="animate-pulse w-2 h-2 bg-blue-600 rounded-full"></div>
                  Streaming updates...
                </div>
              )}
              
              {connectionError && (
                <div className="text-sm text-red-600 bg-red-50 p-2 rounded">
                  {connectionError}
                </div>
              )}
            </div>

            {/* User Input */}
            <div>
              <h3 className="font-semibold mb-2">User Input</h3>
              <div className="space-y-2">
                <div>
                  <label className="text-sm text-gray-600">User ID:</label>
                  <input 
                    type="text"
                    value={selectedUserId}
                    onChange={(e) => setSelectedUserId(e.target.value)}
                    placeholder="Enter user ID (e.g., 12345)"
                    className="w-full mt-1 p-2 border border-gray-300 rounded text-sm"
                    disabled={isStreaming}
                  />
                </div>
                <div>
                  <label className="text-sm text-gray-600">Max Iterations:</label>
                  <input 
                    type="number"
                    min="1"
                    max="10"
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(parseInt(e.target.value) || 1)}
                    className="w-full mt-1 p-2 border border-gray-300 rounded text-sm"
                    disabled={isStreaming}
                  />
                </div>
                <Button 
                  onClick={startNewRecommendation}
                  disabled={!isConnected || isStreaming || !selectedUserId.trim()}
                  className="w-full"
                >
                  Start Recommendation
                </Button>
              </div>
            </div>

            {currentIteration && (
              <>
                <Separator />
                <div>
                  <h3 className="font-semibold mb-2">Session Info</h3>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-gray-600">Current Iteration:</span>
                      <span className="ml-2 font-mono">{currentIteration.iteration_number}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Max Iterations:</span>
                      <span className="ml-2">{currentIteration.recommendation.max_iterations}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Total Iterations:</span>
                      <span className="ml-2">{iterations.length}</span>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold mb-2">Current Precision</h3>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-gray-600">Recommendation:</span>
                      <span className="ml-2 font-mono">{currentIteration.recommendation.precision.toFixed(3)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Evaluation:</span>
                      <span className="ml-2 font-mono">{currentIteration.evaluation.precision.toFixed(3)}</span>
                    </div>
                  </div>
                </div>
              </>
            )}

            <Separator />
            
            <div>
              <h3 className="font-semibold mb-2">Actions</h3>
              <div className="space-y-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full justify-start"
                  onClick={() => setExpandedCards(new Set(iterations.map(i => i.iteration_number)))}
                >
                  Expand All
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full justify-start"
                  onClick={() => setExpandedCards(new Set())}
                >
                  Collapse All
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full justify-start"
                  onClick={() => setIterations([])}
                >
                  Clear Data
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-6">
          <div className="max-w-6xl">
            <h1 className="text-2xl font-bold mb-6">AI Recommendation Evolution</h1>
            
            <ScrollArea className="h-[calc(100vh-120px)]">
              <div className="space-y-4">
                {iterations.map((iteration, index) => (
                  <IterationCard
                    key={iteration.iteration_number}
                    data={iteration}
                    previousData={index > 0 ? iterations[index - 1] : undefined}
                    isExpanded={expandedCards.has(iteration.iteration_number)}
                    onToggleExpand={() => toggleExpanded(iteration.iteration_number)}
                    contentData={contentData}
                  />
                ))}
                
                {iterations.length === 0 && (
                  <div className="text-center py-12 text-gray-500">
                    <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Waiting for iteration data...</p>
                    {!isConnected && (
                      <p className="text-sm mt-2">Please ensure the WebSocket server is running on ws://localhost:8000</p>
                    )}
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IterationVisualizer;
