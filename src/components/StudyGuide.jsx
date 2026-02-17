import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Import study guide content as raw text
import studyGuideContent from '../../aws-ml-specialty-study-guide.md?raw';

const STORAGE_KEY = 'aws-ml-study-checklist';

function getChecklistState() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : {};
  } catch {
    return {};
  }
}

function saveChecklistState(state) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    console.error('Failed to save checklist state:', e);
  }
}

export default function StudyGuide({ onBack }) {
  const [checklistState, setChecklistState] = useState(getChecklistState);
  const [activeSection, setActiveSection] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Parse sections from markdown
  const sections = React.useMemo(() => {
    const lines = studyGuideContent.split('\n');
    const sectionList = [];
    let currentSection = null;
    let currentContent = [];

    for (const line of lines) {
      if (line.startsWith('## ')) {
        if (currentSection) {
          sectionList.push({ ...currentSection, content: currentContent.join('\n') });
        }
        currentSection = { title: line.replace('## ', ''), id: line.replace('## ', '').toLowerCase().replace(/[^a-z0-9]+/g, '-') };
        currentContent = [line];
      } else if (currentSection) {
        currentContent.push(line);
      }
    }
    if (currentSection) {
      sectionList.push({ ...currentSection, content: currentContent.join('\n') });
    }
    return sectionList;
  }, []);

  // Extract checklist items
  const checklistItems = React.useMemo(() => {
    const items = [];
    const regex = /- \[ \] (.+)/g;
    let match;
    while ((match = regex.exec(studyGuideContent)) !== null) {
      items.push({
        id: match[1].substring(0, 50).toLowerCase().replace(/[^a-z0-9]+/g, '-'),
        text: match[1]
      });
    }
    return items;
  }, []);

  const completedCount = Object.values(checklistState).filter(Boolean).length;

  const handleCheckboxChange = (itemId) => {
    const newState = { ...checklistState, [itemId]: !checklistState[itemId] };
    setChecklistState(newState);
    saveChecklistState(newState);
  };

  const handleDownloadPDF = () => {
    const a = document.createElement('a');
    a.href = '/aws-ml-specialty-study-guide.pdf';
    a.download = 'aws-ml-specialty-study-guide.pdf';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Custom renderer for checkboxes in markdown
  const components = {
    // Handle list items with checkboxes (task list items)
    input: ({ type, checked, ...props }) => {
      if (type === 'checkbox') {
        // Find the closest text to identify this checkbox
        return (
          <input
            type="checkbox"
            checked={checked}
            disabled
            className="mr-2 w-4 h-4 rounded border-slate-500 bg-slate-700 text-amber-500 opacity-50"
            {...props}
          />
        );
      }
      return <input type={type} {...props} />;
    },
    li: ({ children, node, ...props }) => {
      return <li className="my-1 text-slate-300" {...props}>{children}</li>;
    },
    // Style tables
    table: ({ children }) => (
      <div className="overflow-x-auto my-4">
        <table className="min-w-full border-collapse text-sm">{children}</table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className="bg-slate-700/50">{children}</thead>
    ),
    th: ({ children }) => (
      <th className="px-3 py-2 text-left text-amber-500 font-semibold border border-slate-600">{children}</th>
    ),
    td: ({ children }) => (
      <td className="px-3 py-2 text-slate-300 border border-slate-600">{children}</td>
    ),
    // Style headings
    h1: ({ children }) => (
      <h1 className="text-2xl sm:text-3xl font-bold text-white mb-4 mt-6">{children}</h1>
    ),
    h2: ({ children }) => (
      <h2 className="text-xl sm:text-2xl font-bold text-amber-500 mb-3 mt-8 pb-2 border-b border-slate-700">{children}</h2>
    ),
    h3: ({ children }) => (
      <h3 className="text-lg sm:text-xl font-semibold text-white mb-2 mt-6">{children}</h3>
    ),
    h4: ({ children }) => (
      <h4 className="text-base sm:text-lg font-semibold text-slate-200 mb-2 mt-4">{children}</h4>
    ),
    // Style paragraphs
    p: ({ children }) => (
      <p className="text-slate-300 mb-3 leading-relaxed">{children}</p>
    ),
    // Style code
    code: ({ inline, children }) => (
      inline ? (
        <code className="px-1.5 py-0.5 bg-slate-700 text-amber-400 rounded text-sm font-mono">{children}</code>
      ) : (
        <code className="block p-4 bg-slate-900 rounded-lg text-sm font-mono text-green-400 overflow-x-auto my-4">{children}</code>
      )
    ),
    pre: ({ children }) => (
      <pre className="bg-slate-900 rounded-lg overflow-x-auto my-4">{children}</pre>
    ),
    // Style blockquotes
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-amber-500 pl-4 my-4 text-slate-400 italic">{children}</blockquote>
    ),
    // Style lists
    ul: ({ children }) => (
      <ul className="list-disc list-inside space-y-1 mb-4 text-slate-300 ml-4">{children}</ul>
    ),
    ol: ({ children }) => (
      <ol className="list-decimal list-inside space-y-1 mb-4 text-slate-300 ml-4">{children}</ol>
    ),
    // Style strong/emphasis
    strong: ({ children }) => (
      <strong className="font-semibold text-white">{children}</strong>
    ),
    em: ({ children }) => (
      <em className="italic text-slate-400">{children}</em>
    ),
    // Style horizontal rule
    hr: () => (
      <hr className="my-8 border-slate-700" />
    ),
    // Style links
    a: ({ children, href }) => (
      <a href={href} className="text-amber-500 hover:text-amber-400 underline" target="_blank" rel="noopener noreferrer">{children}</a>
    ),
  };

  // Filter content based on search
  const filteredSections = searchTerm
    ? sections.filter(s => s.content.toLowerCase().includes(searchTerm.toLowerCase()))
    : sections;

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Header */}
      <div className="sticky top-0 z-20 bg-slate-900/95 backdrop-blur border-b border-slate-700">
        <div className="max-w-5xl mx-auto px-4 py-3">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-3">
              <button
                onClick={onBack}
                className="p-2 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-slate-800"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <h1 className="text-lg sm:text-xl font-bold text-white">Study Guide</h1>
            </div>
            
            <div className="flex items-center gap-2">
              {/* Search */}
              <div className="relative flex-1 sm:flex-initial">
                <input
                  type="text"
                  placeholder="Search..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full sm:w-48 px-3 py-1.5 pl-9 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm placeholder-slate-400 focus:outline-none focus:border-amber-500"
                />
                <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              
              {/* Export button */}
              <button
                onClick={handleDownloadPDF}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500/20 text-amber-500 hover:bg-amber-500/30 transition-colors rounded-lg text-sm font-medium"
                title="Download PDF"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span className="hidden sm:inline">PDF</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 py-6">
        {/* Progress Card */}
        <div className="bg-slate-800 rounded-xl p-4 mb-6 border border-slate-700">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <h2 className="text-white font-semibold mb-1">Practice Checklist Progress</h2>
              <p className="text-slate-400 text-sm">{completedCount} of {checklistItems.length} items completed</p>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex-1 sm:w-32 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-orange-500 to-amber-500 transition-all duration-300"
                  style={{ width: `${checklistItems.length ? (completedCount / checklistItems.length) * 100 : 0}%` }}
                />
              </div>
              <span className="text-amber-500 font-semibold text-sm">
                {checklistItems.length ? Math.round((completedCount / checklistItems.length) * 100) : 0}%
              </span>
            </div>
          </div>
        </div>

        {/* Table of Contents */}
        <div className="bg-slate-800 rounded-xl p-4 mb-6 border border-slate-700">
          <h2 className="text-white font-semibold mb-3">Quick Navigation</h2>
          <div className="flex flex-wrap gap-2">
            {sections.map((section) => (
              <a
                key={section.id}
                href={`#${section.id}`}
                className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                  filteredSections.includes(section)
                    ? 'bg-slate-700 text-slate-200 hover:bg-slate-600'
                    : 'bg-slate-800 text-slate-500'
                }`}
              >
                {section.title}
              </a>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="bg-slate-800 rounded-xl p-4 sm:p-6 border border-slate-700">
          {filteredSections.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-slate-400">No sections match your search.</p>
            </div>
          ) : (
            filteredSections.map((section) => (
              <div key={section.id} id={section.id} className="scroll-mt-20">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={components}
                >
                  {section.content}
                </ReactMarkdown>
              </div>
            ))
          )}
        </div>

        {/* Interactive Checklist Section */}
        <div className="bg-slate-800 rounded-xl p-4 sm:p-6 mt-6 border border-slate-700" id="checklist">
          <h2 className="text-xl font-bold text-amber-500 mb-4 flex items-center gap-2">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
            Practice Checklist
          </h2>
          <p className="text-slate-400 text-sm mb-4">Track your exam preparation progress. Your progress is saved locally.</p>
          
          <div className="space-y-3">
            {checklistItems.map((item) => (
              <label
                key={item.id}
                className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                  checklistState[item.id] 
                    ? 'bg-green-500/10 border border-green-500/30' 
                    : 'bg-slate-700/50 border border-slate-600 hover:border-slate-500'
                }`}
              >
                <input
                  type="checkbox"
                  checked={!!checklistState[item.id]}
                  onChange={() => handleCheckboxChange(item.id)}
                  className="mt-0.5 w-5 h-5 rounded border-slate-500 bg-slate-700 text-amber-500 focus:ring-amber-500 focus:ring-offset-slate-800 cursor-pointer flex-shrink-0"
                />
                <span className={`text-sm sm:text-base ${checklistState[item.id] ? 'line-through text-slate-500' : 'text-slate-300'}`}>
                  {item.text}
                </span>
              </label>
            ))}
          </div>
          
          {completedCount === checklistItems.length && checklistItems.length > 0 && (
            <div className="mt-6 p-4 bg-green-500/10 border border-green-500/30 rounded-xl text-center">
              <div className="text-green-500 font-semibold text-lg mb-1">Congratulations!</div>
              <p className="text-slate-400 text-sm">You've completed all checklist items. You're ready for the exam!</p>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
