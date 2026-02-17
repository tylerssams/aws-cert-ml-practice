import React, { useState } from 'react';
import { getProfiles, createProfile, setActiveProfile, deleteProfile } from '../storage';

export default function ProfileSelect({ onProfileSelected }) {
  const [profiles, setProfilesList] = useState(getProfiles());
  const [isCreating, setIsCreating] = useState(false);
  const [newName, setNewName] = useState('');
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  const handleCreateProfile = (e) => {
    e.preventDefault();
    if (newName.trim()) {
      const profile = createProfile(newName);
      setProfilesList(getProfiles());
      setNewName('');
      setIsCreating(false);
      onProfileSelected(profile);
    }
  };

  const handleSelectProfile = (profile) => {
    setActiveProfile(profile.id);
    onProfileSelected(profile);
  };

  const handleDeleteProfile = (profileId) => {
    deleteProfile(profileId);
    setProfilesList(getProfiles());
    setDeleteConfirm(null);
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4 sm:p-6">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-6 sm:mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 sm:w-16 sm:h-16 bg-gradient-to-br from-orange-500 to-amber-600 rounded-xl sm:rounded-2xl mb-3 sm:mb-4">
            <svg className="w-6 h-6 sm:w-8 sm:h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-white mb-2">AWS ML Specialty Practice</h1>
          <p className="text-slate-400 text-sm sm:text-base">Select or create a profile</p>
        </div>

        {/* Profile List */}
        <div className="bg-slate-800 rounded-xl sm:rounded-2xl border border-slate-700 overflow-hidden">
          {profiles.length > 0 && !isCreating && (
            <div className="p-3 sm:p-4 border-b border-slate-700">
              <h2 className="text-xs sm:text-sm font-medium text-slate-400 uppercase tracking-wide">Your Profiles</h2>
            </div>
          )}

          {/* Existing Profiles */}
          {!isCreating && profiles.map((profile) => (
            <div
              key={profile.id}
              className="group relative"
            >
              {deleteConfirm === profile.id ? (
                <div className="p-3 sm:p-4 bg-red-500/10 border-b border-slate-700">
                  <p className="text-red-400 text-xs sm:text-sm mb-3">Delete "{profile.name}"?</p>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleDeleteProfile(profile.id)}
                      className="px-3 sm:px-4 py-2 bg-red-500 text-white text-xs sm:text-sm rounded-lg hover:bg-red-600 transition-colors"
                    >
                      Delete
                    </button>
                    <button
                      onClick={() => setDeleteConfirm(null)}
                      className="px-3 sm:px-4 py-2 bg-slate-600 text-white text-xs sm:text-sm rounded-lg hover:bg-slate-500 transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  onClick={() => handleSelectProfile(profile)}
                  className="w-full p-3 sm:p-4 text-left hover:bg-slate-700/50 transition-colors border-b border-slate-700 flex items-center justify-between"
                >
                  <div className="flex items-center space-x-2 sm:space-x-3 min-w-0">
                    <div className="w-9 h-9 sm:w-10 sm:h-10 bg-gradient-to-br from-orange-500 to-amber-600 rounded-full flex items-center justify-center flex-shrink-0">
                      <span className="text-white font-semibold text-sm sm:text-base">
                        {profile.name.charAt(0).toUpperCase()}
                      </span>
                    </div>
                    <div className="min-w-0">
                      <div className="text-white font-medium text-sm sm:text-base truncate">{profile.name}</div>
                      <div className="text-slate-500 text-xs sm:text-sm truncate">
                        {formatDate(profile.createdAt)}
                        {Object.keys(profile.attempts).length > 0 && (
                          <span> · {Object.keys(profile.attempts).length} attempts</span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-1 sm:space-x-2 flex-shrink-0 ml-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteConfirm(profile.id);
                      }}
                      className="sm:opacity-0 sm:group-hover:opacity-100 p-1.5 sm:p-2 text-slate-400 hover:text-red-400 transition-all"
                    >
                      <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                    <svg className="w-4 h-4 sm:w-5 sm:h-5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </button>
              )}
            </div>
          ))}

          {/* Create New Profile Form */}
          {isCreating ? (
            <form onSubmit={handleCreateProfile} className="p-3 sm:p-4">
              <label className="block text-xs sm:text-sm font-medium text-slate-400 mb-2">
                Profile Name
              </label>
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Enter your name..."
                autoFocus
                className="w-full px-3 sm:px-4 py-2.5 sm:py-3 bg-slate-700 border border-slate-600 rounded-lg sm:rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 transition-colors text-sm sm:text-base"
              />
              <div className="flex space-x-2 sm:space-x-3 mt-3 sm:mt-4">
                <button
                  type="submit"
                  disabled={!newName.trim()}
                  className="flex-1 py-2.5 sm:py-3 bg-gradient-to-r from-orange-500 to-amber-600 text-white font-semibold rounded-lg sm:rounded-xl hover:from-orange-600 hover:to-amber-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all text-sm sm:text-base"
                >
                  Create
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setIsCreating(false);
                    setNewName('');
                  }}
                  className="px-4 sm:px-6 py-2.5 sm:py-3 bg-slate-700 text-white rounded-lg sm:rounded-xl hover:bg-slate-600 transition-colors text-sm sm:text-base"
                >
                  Cancel
                </button>
              </div>
            </form>
          ) : (
            <button
              onClick={() => setIsCreating(true)}
              className="w-full p-3 sm:p-4 text-left hover:bg-slate-700/50 transition-colors flex items-center space-x-2 sm:space-x-3 text-amber-500"
            >
              <div className="w-9 h-9 sm:w-10 sm:h-10 border-2 border-dashed border-amber-500/50 rounded-full flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </div>
              <span className="font-medium text-sm sm:text-base">Create New Profile</span>
            </button>
          )}
        </div>

        {/* Footer */}
        <p className="text-center text-slate-500 text-xs sm:text-sm mt-4 sm:mt-6">
          Progress saved locally in this browser
        </p>
      </div>
    </div>
  );
}
