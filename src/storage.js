// LocalStorage key for all app data
const STORAGE_KEY = 'aws-ml-practice';
const DATA_VERSION = 1;

// Generate unique IDs
export const generateId = () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

// Initialize empty data structure
const getEmptyData = () => ({
  version: DATA_VERSION,
  activeProfileId: null,
  profiles: {}
});

// Load all data from localStorage
export const loadData = () => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return getEmptyData();
    }
    const data = JSON.parse(stored);
    // Handle version migrations here if needed in the future
    if (!data.version) {
      data.version = DATA_VERSION;
    }
    return data;
  } catch (error) {
    console.error('Failed to load data from localStorage:', error);
    return getEmptyData();
  }
};

// Save all data to localStorage
export const saveData = (data) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    return true;
  } catch (error) {
    console.error('Failed to save data to localStorage:', error);
    // Handle quota exceeded
    if (error.name === 'QuotaExceededError') {
      alert('Local storage is full. Please delete some profiles or attempts.');
    }
    return false;
  }
};

// ============ Profile Operations ============

export const getProfiles = () => {
  const data = loadData();
  return Object.values(data.profiles).sort((a, b) => 
    new Date(b.createdAt) - new Date(a.createdAt)
  );
};

export const getActiveProfile = () => {
  const data = loadData();
  if (!data.activeProfileId || !data.profiles[data.activeProfileId]) {
    return null;
  }
  return data.profiles[data.activeProfileId];
};

export const createProfile = (name) => {
  const data = loadData();
  const id = `profile-${generateId()}`;
  const profile = {
    id,
    name: name.trim(),
    createdAt: new Date().toISOString(),
    attempts: {}
  };
  data.profiles[id] = profile;
  data.activeProfileId = id;
  saveData(data);
  return profile;
};

export const setActiveProfile = (profileId) => {
  const data = loadData();
  if (data.profiles[profileId]) {
    data.activeProfileId = profileId;
    saveData(data);
    return true;
  }
  return false;
};

export const deleteProfile = (profileId) => {
  const data = loadData();
  if (data.profiles[profileId]) {
    delete data.profiles[profileId];
    if (data.activeProfileId === profileId) {
      data.activeProfileId = null;
    }
    saveData(data);
    return true;
  }
  return false;
};

// ============ Attempt Operations ============

export const getAttempts = (profileId, testId = null) => {
  const data = loadData();
  const profile = data.profiles[profileId];
  if (!profile) return [];
  
  let attempts = Object.values(profile.attempts);
  if (testId) {
    attempts = attempts.filter(a => a.testId === testId);
  }
  return attempts.sort((a, b) => new Date(b.startedAt) - new Date(a.startedAt));
};

export const getInProgressAttempt = (profileId, testId) => {
  const attempts = getAttempts(profileId, testId);
  return attempts.find(a => a.status === 'in-progress') || null;
};

export const createAttempt = (profileId, testId) => {
  const data = loadData();
  const profile = data.profiles[profileId];
  if (!profile) return null;
  
  const id = `attempt-${generateId()}`;
  const attempt = {
    id,
    testId,
    status: 'in-progress',
    startedAt: new Date().toISOString(),
    completedAt: null,
    answers: {},
    currentQuestion: 0,
    elapsedTime: 0,
    score: null,
    percentage: null,
    passed: null
  };
  profile.attempts[id] = attempt;
  saveData(data);
  return attempt;
};

export const updateAttemptProgress = (profileId, attemptId, { answers, currentQuestion, elapsedTime }) => {
  const data = loadData();
  const profile = data.profiles[profileId];
  if (!profile || !profile.attempts[attemptId]) return false;
  
  const attempt = profile.attempts[attemptId];
  if (answers !== undefined) attempt.answers = answers;
  if (currentQuestion !== undefined) attempt.currentQuestion = currentQuestion;
  if (elapsedTime !== undefined) attempt.elapsedTime = elapsedTime;
  
  saveData(data);
  return true;
};

export const completeAttempt = (profileId, attemptId, { score, percentage, passed, finalTime }) => {
  const data = loadData();
  const profile = data.profiles[profileId];
  if (!profile || !profile.attempts[attemptId]) return false;
  
  const attempt = profile.attempts[attemptId];
  attempt.status = 'completed';
  attempt.completedAt = new Date().toISOString();
  attempt.score = score;
  attempt.percentage = percentage;
  attempt.passed = passed;
  attempt.elapsedTime = finalTime;
  
  saveData(data);
  return true;
};

export const deleteAttempt = (profileId, attemptId) => {
  const data = loadData();
  const profile = data.profiles[profileId];
  if (!profile || !profile.attempts[attemptId]) return false;
  
  delete profile.attempts[attemptId];
  saveData(data);
  return true;
};

export const getAttemptById = (profileId, attemptId) => {
  const data = loadData();
  const profile = data.profiles[profileId];
  if (!profile || !profile.attempts[attemptId]) return null;
  return profile.attempts[attemptId];
};

// ============ Export Functions ============

export const exportAsJSON = (attempt, testData, questions) => {
  const exportData = {
    exportDate: new Date().toISOString(),
    testName: testData.name,
    attempt: {
      ...attempt,
      questionDetails: questions.map(q => ({
        id: q.id,
        domain: q.domain,
        question: q.question,
        userAnswer: attempt.answers[q.id] || null,
        correctAnswer: q.correct,
        isCorrect: attempt.answers[q.id] === q.correct,
        options: q.options.map(o => ({
          id: o.id,
          text: o.text,
          explanation: q.explanations[o.id]
        }))
      }))
    }
  };
  
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `AWS-ML-${testData.name.replace(/\s+/g, '-')}-${new Date().toISOString().split('T')[0]}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export const exportAsCSV = (attempt, testData, questions) => {
  const rows = [
    ['Question #', 'Domain', 'Question', 'Your Answer', 'Correct Answer', 'Result']
  ];
  
  questions.forEach((q, idx) => {
    const userAnswer = attempt.answers[q.id] || 'Not answered';
    const isCorrect = userAnswer === q.correct;
    rows.push([
      idx + 1,
      q.domain,
      `"${q.question.replace(/"/g, '""')}"`,
      userAnswer,
      q.correct,
      isCorrect ? 'Correct' : 'Incorrect'
    ]);
  });
  
  // Add summary row
  rows.push([]);
  rows.push(['Summary']);
  rows.push(['Score', `${attempt.score}/${questions.length}`]);
  rows.push(['Percentage', `${attempt.percentage}%`]);
  rows.push(['Status', attempt.passed ? 'PASSED' : 'NOT PASSED']);
  
  const csv = rows.map(row => row.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `AWS-ML-${testData.name.replace(/\s+/g, '-')}-${new Date().toISOString().split('T')[0]}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export const exportAsHTML = (attempt, testData, questions, domainScores) => {
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AWS ML Specialty - ${testData.name} Results</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }
    .container { max-width: 900px; margin: 0 auto; padding: 20px; }
    .header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }
    .header h1 { font-size: 24px; margin-bottom: 5px; }
    .header p { opacity: 0.8; }
    .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
    .stat { background: white; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stat-value { font-size: 28px; font-weight: bold; }
    .stat-label { font-size: 12px; color: #666; text-transform: uppercase; }
    .passed { color: #22c55e; }
    .failed { color: #ef4444; }
    .section { background: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .section h2 { font-size: 18px; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #e5e7eb; }
    .domain-bar { margin-bottom: 15px; }
    .domain-header { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px; }
    .domain-progress { height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }
    .domain-fill { height: 100%; border-radius: 4px; }
    .question { border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
    .question-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .question-num { font-weight: bold; }
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .correct { background: #dcfce7; color: #166534; }
    .incorrect { background: #fee2e2; color: #991b1b; }
    .domain-badge { background: #fef3c7; color: #92400e; }
    .question-text { margin-bottom: 15px; }
    .options { list-style: none; }
    .option { padding: 8px 12px; margin-bottom: 8px; border-radius: 6px; display: flex; }
    .option.user-correct { background: #dcfce7; border: 1px solid #22c55e; }
    .option.user-wrong { background: #fee2e2; border: 1px solid #ef4444; }
    .option.correct-answer { background: #dcfce7; border: 1px solid #22c55e; }
    .option-id { font-weight: bold; margin-right: 10px; min-width: 20px; }
    .explanation { margin-top: 10px; padding: 10px; background: #f8fafc; border-radius: 6px; font-size: 13px; color: #64748b; }
    @media print { body { background: white; } .container { padding: 0; } .section, .stat { box-shadow: none; border: 1px solid #e5e7eb; } }
    @media (max-width: 600px) { .summary { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>AWS Machine Learning Specialty</h1>
      <p>${testData.name} - ${new Date(attempt.completedAt || attempt.startedAt).toLocaleString()}</p>
    </div>
    
    <div class="summary">
      <div class="stat">
        <div class="stat-value ${attempt.passed ? 'passed' : 'failed'}">${attempt.passed ? 'PASSED' : 'FAILED'}</div>
        <div class="stat-label">Status</div>
      </div>
      <div class="stat">
        <div class="stat-value">${attempt.score}/${questions.length}</div>
        <div class="stat-label">Correct</div>
      </div>
      <div class="stat">
        <div class="stat-value ${attempt.passed ? 'passed' : 'failed'}">${attempt.percentage}%</div>
        <div class="stat-label">Score</div>
      </div>
      <div class="stat">
        <div class="stat-value">${formatTime(attempt.elapsedTime)}</div>
        <div class="stat-label">Time</div>
      </div>
    </div>
    
    <div class="section">
      <h2>Domain Breakdown</h2>
      ${Object.entries(domainScores).map(([domain, scores]) => {
        const pct = Math.round((scores.correct / scores.total) * 100);
        return `
        <div class="domain-bar">
          <div class="domain-header">
            <span>${domain}</span>
            <span class="${pct >= 75 ? 'passed' : 'failed'}">${scores.correct}/${scores.total} (${pct}%)</span>
          </div>
          <div class="domain-progress">
            <div class="domain-fill" style="width: ${pct}%; background: ${pct >= 75 ? '#22c55e' : '#ef4444'}"></div>
          </div>
        </div>`;
      }).join('')}
    </div>
    
    <div class="section">
      <h2>Question Review</h2>
      ${questions.map((q, idx) => {
        const userAnswer = attempt.answers[q.id];
        const isCorrect = userAnswer === q.correct;
        return `
        <div class="question">
          <div class="question-header">
            <span class="question-num">Question ${idx + 1}</span>
            <span>
              <span class="badge domain-badge">${q.domain}</span>
              <span class="badge ${isCorrect ? 'correct' : 'incorrect'}">${isCorrect ? 'Correct' : 'Incorrect'}</span>
            </span>
          </div>
          <div class="question-text">${q.question}</div>
          <ul class="options">
            ${q.options.map(opt => {
              let className = '';
              if (opt.id === q.correct && opt.id === userAnswer) className = 'user-correct';
              else if (opt.id === userAnswer && opt.id !== q.correct) className = 'user-wrong';
              else if (opt.id === q.correct) className = 'correct-answer';
              return `
              <li class="option ${className}">
                <span class="option-id">${opt.id}.</span>
                <span>${opt.text}</span>
              </li>`;
            }).join('')}
          </ul>
          <div class="explanation">
            <strong>Explanation:</strong> ${q.explanations[q.correct]}
          </div>
        </div>`;
      }).join('')}
    </div>
  </div>
</body>
</html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  window.open(url, '_blank');
};
