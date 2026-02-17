import React, { useState } from 'react';
import { getAttempts, getInProgressAttempt, deleteAttempt } from '../storage';

export default function TestDashboard({ 
  profile, 
  testBank, 
  onStartTest, 
  onResumeAttempt, 
  onViewResults,
  onSwitchProfile 
}) {
  const [expandedTest, setExpandedTest] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    });
  };

  const handleDeleteAttempt = (attemptId) => {
    deleteAttempt(profile.id, attemptId);
    setDeleteConfirm(null);
    // Force re-render by toggling expanded state
    setExpandedTest(prev => prev);
  };

  return (
    <div className="min-h-screen bg-slate-900 p-6">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">Practice Exams</h1>
            <p className="text-slate-400">Choose an exam to practice</p>
          </div>
          <button
            onClick={onSwitchProfile}
            className="flex items-center space-x-2 px-4 py-2 bg-slate-800 border border-slate-700 rounded-xl hover:bg-slate-700 transition-colors"
          >
            <div className="w-8 h-8 bg-gradient-to-br from-orange-500 to-amber-600 rounded-full flex items-center justify-center">
              <span className="text-white font-semibold text-sm">
                {profile.name.charAt(0).toUpperCase()}
              </span>
            </div>
            <span className="text-white font-medium">{profile.name}</span>
            <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
            </svg>
          </button>
        </div>

        {/* Test Cards */}
        <div className="space-y-4">
          {Object.entries(testBank).map(([testId, test]) => {
            const inProgress = getInProgressAttempt(profile.id, testId);
            const completedAttempts = getAttempts(profile.id, testId).filter(a => a.status === 'completed');
            const isExpanded = expandedTest === testId;
            
            return (
              <div
                key={testId}
                className="bg-slate-800 rounded-2xl border border-slate-700 overflow-hidden"
              >
                {/* Test Info */}
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h2 className="text-xl font-semibold text-white mb-1">{test.name}</h2>
                      <p className="text-slate-400">{test.description}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-amber-500">{test.questions.length}</div>
                      <div className="text-slate-500 text-sm">Questions</div>
                    </div>
                  </div>

                  {/* In-Progress Banner */}
                  {inProgress && (
                    <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 mb-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="w-10 h-10 bg-amber-500/20 rounded-full flex items-center justify-center">
                            <svg className="w-5 h-5 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          </div>
                          <div>
                            <div className="text-amber-500 font-medium">In Progress</div>
                            <div className="text-slate-400 text-sm">
                              {Object.keys(inProgress.answers).length}/{test.questions.length} answered · {formatTime(inProgress.elapsedTime)} elapsed
                            </div>
                          </div>
                        </div>
                        <button
                          onClick={() => onResumeAttempt(testId, inProgress)}
                          className="px-4 py-2 bg-amber-500 text-white font-medium rounded-lg hover:bg-amber-600 transition-colors"
                        >
                          Continue
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={() => onStartTest(testId)}
                      className="flex-1 py-3 bg-gradient-to-r from-orange-500 to-amber-600 text-white font-semibold rounded-xl hover:from-orange-600 hover:to-amber-700 transition-all shadow-lg"
                    >
                      {inProgress ? 'Start New Attempt' : 'Take Exam'}
                    </button>
                    {completedAttempts.length > 0 && (
                      <button
                        onClick={() => setExpandedTest(isExpanded ? null : testId)}
                        className="px-4 py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition-colors flex items-center space-x-2"
                      >
                        <span>History ({completedAttempts.length})</span>
                        <svg 
                          className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
                          fill="none" 
                          viewBox="0 0 24 24" 
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>

                {/* Attempt History (Expandable) */}
                {isExpanded && completedAttempts.length > 0 && (
                  <div className="border-t border-slate-700 bg-slate-850">
                    <div className="p-4">
                      <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wide mb-3">
                        Past Attempts
                      </h3>
                      <div className="space-y-2">
                        {completedAttempts.map((attempt) => (
                          <div key={attempt.id}>
                            {deleteConfirm === attempt.id ? (
                              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                                <p className="text-red-400 text-sm mb-2">Delete this attempt?</p>
                                <div className="flex space-x-2">
                                  <button
                                    onClick={() => handleDeleteAttempt(attempt.id)}
                                    className="px-3 py-1.5 bg-red-500 text-white text-sm rounded-lg hover:bg-red-600 transition-colors"
                                  >
                                    Delete
                                  </button>
                                  <button
                                    onClick={() => setDeleteConfirm(null)}
                                    className="px-3 py-1.5 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-500 transition-colors"
                                  >
                                    Cancel
                                  </button>
                                </div>
                              </div>
                            ) : (
                              <div
                                className="group flex items-center justify-between p-3 bg-slate-700/50 rounded-lg hover:bg-slate-700 transition-colors cursor-pointer"
                                onClick={() => onViewResults(testId, attempt)}
                              >
                                <div className="flex items-center space-x-4">
                                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                                    attempt.passed ? 'bg-green-500/20' : 'bg-red-500/20'
                                  }`}>
                                    {attempt.passed ? (
                                      <svg className="w-5 h-5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                      </svg>
                                    ) : (
                                      <svg className="w-5 h-5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                      </svg>
                                    )}
                                  </div>
                                  <div>
                                    <div className="text-white font-medium">
                                      {attempt.score}/{test.questions.length} ({attempt.percentage}%)
                                    </div>
                                    <div className="text-slate-400 text-sm">
                                      {formatDate(attempt.completedAt)} · {formatTime(attempt.elapsedTime)}
                                    </div>
                                  </div>
                                </div>
                                <div className="flex items-center space-x-2">
                                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                                    attempt.passed 
                                      ? 'bg-green-500/20 text-green-500' 
                                      : 'bg-red-500/20 text-red-500'
                                  }`}>
                                    {attempt.passed ? 'PASSED' : 'FAILED'}
                                  </span>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setDeleteConfirm(attempt.id);
                                    }}
                                    className="opacity-0 group-hover:opacity-100 p-1.5 text-slate-400 hover:text-red-400 transition-all"
                                  >
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                  </button>
                                  <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                  </svg>
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Stats Summary */}
        {(() => {
          const allAttempts = Object.keys(testBank).flatMap(testId => 
            getAttempts(profile.id, testId).filter(a => a.status === 'completed')
          );
          if (allAttempts.length === 0) return null;
          
          const passedCount = allAttempts.filter(a => a.passed).length;
          const avgScore = Math.round(allAttempts.reduce((sum, a) => sum + a.percentage, 0) / allAttempts.length);
          
          return (
            <div className="mt-8 bg-slate-800 rounded-2xl border border-slate-700 p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Your Stats</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">{allAttempts.length}</div>
                  <div className="text-slate-400 text-sm">Total Attempts</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-500">{passedCount}</div>
                  <div className="text-slate-400 text-sm">Passed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-amber-500">{avgScore}%</div>
                  <div className="text-slate-400 text-sm">Avg Score</div>
                </div>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
