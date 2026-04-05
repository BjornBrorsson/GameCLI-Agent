import { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

const PROVIDERS = [
  { id: 'gemini_cli', label: 'Gemini CLI (Free)', needsKey: false },
  { id: 'gemini_api', label: 'Gemini API', needsKey: true },
  { id: 'openrouter', label: 'OpenRouter', needsKey: true },
];

const ROLES = [
  { id: 'gamer', label: 'Gamer', desc: 'Play to win with smart strategy' },
  { id: 'reviewer', label: 'Reviewer', desc: 'Evaluate game design, UX, and mechanics' },
  { id: 'tester', label: 'QA Tester', desc: 'Hunt for bugs, glitches, and edge cases' },
  { id: 'speedrunner', label: 'Speedrunner', desc: 'Complete the game as fast as possible' },
];

function App() {
  const [provider, setProvider] = useState('gemini_cli');
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('gemini-2.5-flash');
  const [customModel, setCustomModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);  // [{id, name}]
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState('');
  const [sources, setSources] = useState({ monitors: [], windows: [] });
  const [targetType, setTargetType] = useState('monitor');
  const [targetName, setTargetName] = useState('Monitor 1');
  const [instructions, setInstructions] = useState('');
  const [role, setRole] = useState('gamer');
  const [useGrounding, setUseGrounding] = useState(false);
  const [groundingModel, setGroundingModel] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [logs, setLogs] = useState([]);
  const [previewSrc, setPreviewSrc] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [sessionCost, setSessionCost] = useState(null);
  const [isRecordingMacro, setIsRecordingMacro] = useState(false);
  const [macroNameInput, setMacroNameInput] = useState('');
  const [macrosList, setMacrosList] = useState([]);
  
  const wsRef = useRef(null);
  const logsEndRef = useRef(null);
  const previewIntervalRef = useRef(null);
  const costIntervalRef = useRef(null);
  const providerInfo = PROVIDERS.find(p => p.id === provider);

  useEffect(() => {
    fetchSources();
    fetchInstructions();
    checkStatus();
    connectWebSocket();
    fetchMacros();
    checkResumableSession();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (previewIntervalRef.current) clearInterval(previewIntervalRef.current);
      if (costIntervalRef.current) clearInterval(costIntervalRef.current);
    };
  }, []);

  // Fetch models when provider or API key changes (debounced for key typing)
  useEffect(() => {
    const info = PROVIDERS.find(p => p.id === provider);
    if (info?.needsKey && !apiKey) {
      setAvailableModels([]);
      setModelsError('');
      return;
    }
    const timer = setTimeout(() => fetchModels(provider, apiKey), info?.needsKey ? 600 : 0);
    return () => clearTimeout(timer);
  }, [provider, apiKey]);

  const fetchModels = async (prov, key) => {
    setModelsLoading(true);
    setModelsError('');
    try {
      const res = await fetch('http://localhost:8000/api/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: prov, api_key: key || '' })
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const models = data.models || [];
      setAvailableModels(models);
      if (data.default && models.some(m => m.id === data.default)) {
        setSelectedModel(data.default);
      } else if (models.length > 0) {
        setSelectedModel(models[0].id);
      }
      setCustomModel('');
    } catch (e) {
      console.error('Failed to fetch models', e);
      setModelsError(e.message);
      setAvailableModels([]);
    } finally {
      setModelsLoading(false);
    }
  };

  // Poll cost when running with a paid provider
  useEffect(() => {
    if (costIntervalRef.current) clearInterval(costIntervalRef.current);
    if (isRunning && provider !== 'gemini_cli') {
      const fetchCost = async () => {
        try {
          const res = await fetch('http://localhost:8000/api/cost');
          const data = await res.json();
          setSessionCost(data);
        } catch (e) {
          console.error('Failed to fetch cost', e);
        }
      };
      fetchCost();
      costIntervalRef.current = setInterval(fetchCost, 5000);
    } else {
      setSessionCost(null);
    }
    return () => { if (costIntervalRef.current) clearInterval(costIntervalRef.current); };
  }, [isRunning, provider]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Auto-refresh preview when target changes or when running
  useEffect(() => {
    fetchPreview();
    if (previewIntervalRef.current) clearInterval(previewIntervalRef.current);
    const interval = isRunning ? 3000 : 5000;
    previewIntervalRef.current = setInterval(fetchPreview, interval);
    return () => {
      if (previewIntervalRef.current) clearInterval(previewIntervalRef.current);
    };
  }, [targetType, targetName, isRunning]);

  const fetchPreview = useCallback(async () => {
    if (!targetName) return;
    setPreviewLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_type: targetType, target_name: targetName })
      });
      const data = await res.json();
      if (data.image) {
        setPreviewSrc(`data:image/jpeg;base64,${data.image}`);
      }
    } catch (e) {
      console.error("Failed to fetch preview", e);
    } finally {
      setPreviewLoading(false);
    }
  }, [targetType, targetName]);

  const fetchSources = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/sources');
      const data = await res.json();
      setSources(data);
      if (data.monitors.length > 0 && targetType === 'monitor') {
        setTargetName(data.monitors[0]);
      } else if (data.windows.length > 0 && targetType === 'window') {
        setTargetName(data.windows[0]);
      }
    } catch (e) {
      console.error("Failed to fetch sources", e);
    }
  };

  const fetchInstructions = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/instructions');
      const data = await res.json();
      setInstructions(data.instructions);
    } catch (e) {
      console.error("Failed to fetch instructions", e);
    }
  };

  const checkResumableSession = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/session');
      const data = await res.json();
      if (data.resumable && data.session) {
        const s = data.session;
        const age = Math.round((Date.now() / 1000 - s.saved_at) / 60);
        const resume = window.confirm(
          `A previous session was interrupted ${age} minute(s) ago.\n\n` +
          `Target: ${s.target_name}\nModel: ${s.model_name}\nRole: ${s.role}\nStep: ${s.step}\n\n` +
          `Restore settings and continue?`
        );
        if (resume) {
          setTargetType(s.target_type);
          setTargetName(s.target_name);
          setSelectedModel(s.model_name);
          setRole(s.role);
          setInstructions(s.game_instructions);
          setProvider(s.provider || 'gemini_cli');
          setUseGrounding(s.use_grounding || false);
          setGroundingModel(s.grounding_model || '');
        } else {
          await fetch('http://localhost:8000/api/session/clear', { method: 'POST' });
        }
      }
    } catch (e) {}
  };

  const checkStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/status');
      const data = await res.json();
      setIsRunning(data.is_running);
      setIsPaused(data.is_paused || false);
      setIsRecordingMacro(data.is_recording_macro || false);
    } catch (e) {
      console.error('Failed to check status', e);
    }
  };

  const fetchMacros = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/macros');
      const data = await res.json();
      setMacrosList(data.macros || []);
    } catch (e) {
      console.error("Failed to fetch macros", e);
    }
  };

  const handleToggleMacroRecording = async () => {
    if (isRecordingMacro) {
      try {
        const res = await fetch('http://localhost:8000/api/macros/stop_recording', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
          setIsRecordingMacro(false);
          setMacroNameInput('');
          fetchMacros(); // Refresh list
        }
      } catch (e) {
        console.error("Failed to stop macro recording", e);
      }
    } else {
      if (!macroNameInput.trim()) {
        alert('Please enter a macro name before recording.');
        return;
      }
      try {
        const res = await fetch('http://localhost:8000/api/macros/start_recording', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ macro_name: macroNameInput.trim() })
        });
        const data = await res.json();
        if (data.status === 'success') {
          setIsRecordingMacro(true);
        } else {
          alert('Failed to start recording: ' + data.message);
        }
      } catch (e) {
        console.error("Failed to start macro recording", e);
      }
    }
  };

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/ws/logs');
    ws.onmessage = (event) => {
      const msg = event.data;
      // Strip timestamp prefix for classification
      const stripped = msg.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, '');
      // Detect PAUSED state from backend
      if (stripped.startsWith('PAUSED:')) {
        setIsPaused(true);
      }
      if (stripped.startsWith('STATUS:') && stripped.includes('resumed')) {
        setIsPaused(false);
      }
      // STATUS/THINKING messages go to browser console only
      if (stripped.startsWith('STATUS:') || stripped.startsWith('THINKING:')) {
        console.log(msg);
        // Still add THINKING to logs as a subtle indicator
        if (stripped.startsWith('THINKING:')) {
          setLogs((prev) => [...prev, msg]);
        }
        return;
      }
      setLogs((prev) => [...prev, msg]);
    };
    ws.onclose = () => {
      setTimeout(connectWebSocket, 3000);
    };
    wsRef.current = ws;
  };

  const handleStart = async () => {
    try {
      await fetch('http://localhost:8000/api/instructions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instructions })
      });

      const modelToUse = customModel.trim() || selectedModel;
      const res = await fetch('http://localhost:8000/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: apiKey || "local-cli",
          target_type: targetType,
          target_name: targetName,
          model_name: modelToUse,
          instructions: instructions,
          provider: provider,
          role: role,
          use_grounding: useGrounding,
          grounding_model: groundingModel
        })
      });
      
      const data = await res.json();
      if (data.status === 'success') {
        setIsRunning(true);
        setLogs([]);
      } else {
        setLogs(prev => [...prev, `ERROR: ${data.message}`]);
      }
    } catch (e) {
      setLogs(prev => [...prev, `ERROR: Could not connect to backend: ${e.message}`]);
    }
  };

  const handleStop = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/stop', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        setIsRunning(false);
        setIsPaused(false);
      }
    } catch (e) {
      setLogs(prev => [...prev, `ERROR: Could not connect to backend: ${e.message}`]);
    }
  };

  const handleResume = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/resume', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        setIsPaused(false);
      }
    } catch (e) {
      setLogs(prev => [...prev, `ERROR: Could not connect to backend: ${e.message}`]);
    }
  };

  const handleAbort = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/abort', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        setIsRunning(false);
        setIsPaused(false);
      }
    } catch (e) {
      setLogs(prev => [...prev, `ERROR: Could not connect to backend: ${e.message}`]);
    }
  };

  const getLogClass = (log) => {
    const stripped = log.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, '');
    if (stripped.startsWith('ERROR:')) return 'log-error';
    if (stripped.startsWith('WARNING:')) return 'log-warning';
    if (stripped.startsWith('THINKING:')) return 'log-thinking';
    if (stripped.startsWith('NARRATION:')) return 'log-narration';
    if (stripped.startsWith('ACTION:')) return 'log-action';
    if (stripped.startsWith('EXECUTING:')) return 'log-executing';
    if (stripped.startsWith('PAUSED:')) return 'log-paused';
    return '';
  };

  const formatLog = (log) => {
    const stripped = log.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, '');
    const tsMatch = log.match(/^\[(\d{2}:\d{2}:\d{2})\]/);
    const ts = tsMatch ? tsMatch[1] : '';
    
    if (stripped.startsWith('NARRATION:')) {
      const text = stripped.replace('NARRATION: ', '');
      return <><span className="log-ts">{ts}</span> <span className="log-tag tag-narration">THOUGHT</span> {text}</>;
    }
    if (stripped.startsWith('ACTION:')) {
      const text = stripped.replace('ACTION: ', '');
      return <><span className="log-ts">{ts}</span> <span className="log-tag tag-action">ACTION</span> {text}</>;
    }
    if (stripped.startsWith('THINKING:')) {
      const text = stripped.replace('THINKING: ', '');
      return <><span className="log-ts">{ts}</span> <span className="log-tag tag-thinking">...</span> {text}</>;
    }
    if (stripped.startsWith('EXECUTING:')) {
      const text = stripped.replace('EXECUTING: ', '');
      return <><span className="log-ts">{ts}</span> <span className="log-tag tag-executing">EXEC</span> {text}</>;
    }
    if (stripped.startsWith('PAUSED:')) {
      const text = stripped.replace('PAUSED: ', '');
      return <><span className="log-ts">{ts}</span> <span className="log-tag tag-paused">PAUSED</span> {text}</>;
    }
    if (stripped.startsWith('WARNING:')) {
      const text = stripped.replace('WARNING: ', '');
      return <><span className="log-ts">{ts}</span> <span className="log-tag tag-warning">WARN</span> {text}</>;
    }
    if (stripped.startsWith('ERROR:')) {
      return <><span className="log-ts">{ts}</span> <span className="log-tag tag-error">ERROR</span> {stripped.replace('ERROR: ', '')}</>;
    }
    return log;
  };

  // ─── CONFIG VIEW (agent not running) ───
  if (!isRunning) {
    return (
      <>
        <div className="header-flex">
          <h1>Game AI Agent</h1>
          <div className="status-badge status-stopped">
            <div className="dot"></div>
            Agent Stopped
          </div>
        </div>
        
        <div className="dashboard-grid">
          <div className="left-col">
            <div className="card">
              <h2>Configuration</h2>
              
              <label>AI Provider</label>
              <select value={provider} onChange={e => {
                setProvider(e.target.value);
                setCustomModel('');
              }}>
                {PROVIDERS.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
              </select>

              {providerInfo?.needsKey && (
                <>
                  <label>API Key</label>
                  <input
                    type="password"
                    autoComplete="off"
                    value={apiKey}
                    onChange={e => setApiKey(e.target.value)}
                    placeholder={provider === 'openrouter' ? 'sk-or-...' : 'AIza...'}
                  />
                </>
              )}

              <label>
                AI Model
                {modelsLoading && <span style={{marginLeft:8, fontSize:'0.7rem', color:'var(--text-secondary)'}}>loading...</span>}
              </label>
              {modelsError && (
                <div style={{fontSize:'0.75rem', color:'#f87171', marginBottom:'0.5rem'}}>
                  Failed to load models: {modelsError}
                </div>
              )}
              <select
                value={selectedModel}
                onChange={e => { setSelectedModel(e.target.value); setCustomModel(''); }}
                disabled={modelsLoading || availableModels.length === 0}
              >
                {availableModels.length === 0 && !modelsLoading && (
                  <option value="">— enter API key to load models —</option>
                )}
                {availableModels.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
              {provider !== 'gemini_cli' && (
                <>
                  <label>Or enter custom model ID</label>
                  <input
                    type="text"
                    value={customModel}
                    onChange={e => setCustomModel(e.target.value)}
                    placeholder={provider === 'openrouter' ? 'e.g. meta-llama/llama-4-maverick' : 'e.g. gemini-2.5-flash'}
                  />
                </>
              )}

              <label>Target Type</label>
              <select 
                value={targetType} 
                onChange={e => {
                  setTargetType(e.target.value);
                  if (e.target.value === 'monitor' && sources.monitors.length) setTargetName(sources.monitors[0]);
                  if (e.target.value === 'window' && sources.windows.length) setTargetName(sources.windows[0]);
                }}
              >
                <option value="monitor">Monitor</option>
                <option value="window">Window</option>
              </select>

              <label>
                Target Name 
                <button className="btn-inline" onClick={fetchSources}>Refresh</button>
              </label>
              <select value={targetName} onChange={e => setTargetName(e.target.value)}>
                {targetType === 'monitor' 
                  ? sources.monitors.map((m, i) => <option key={`mon-${i}`} value={m}>{m}</option>)
                  : sources.windows.map((w, i) => <option key={`win-${i}`} value={w}>{w}</option>)
                }
              </select>

              <label>Agent Role</label>
              <select value={role} onChange={e => setRole(e.target.value)}>
                {ROLES.map(r => <option key={r.id} value={r.id}>{r.label} — {r.desc}</option>)}
              </select>

              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input type="checkbox" checked={useGrounding} onChange={e => setUseGrounding(e.target.checked)} />
                Visual Grounding
                <span style={{ fontSize: '0.8em', opacity: 0.7 }}>(UI element detection — extra LLM call per step, improves click accuracy)</span>
              </label>

              {useGrounding && (
                <>
                  <label>Grounding Model <span style={{ fontSize: '0.8em', opacity: 0.7 }}>(leave blank to use main model)</span></label>
                  <input
                    type="text"
                    value={groundingModel}
                    onChange={e => setGroundingModel(e.target.value)}
                    placeholder="e.g. gemini-2.0-flash (fast/cheap)"
                  />
                </>
              )}

              <label>Game Instructions (Markdown)</label>
              <textarea 
                value={instructions} 
                onChange={e => setInstructions(e.target.value)} 
                placeholder="Give the agent rules and a goal..."
              />

              <div className="button-group">
                <button
                  className="btn-start"
                  onClick={handleStart}
                  disabled={(providerInfo?.needsKey && !apiKey) || (!customModel.trim() && !selectedModel) || isRecordingMacro}
                >
                  Start Agent
                </button>
              </div>
              {providerInfo?.needsKey && (
                <div style={{marginTop:'0.75rem', fontSize:'0.75rem', color:'var(--text-secondary)'}}>
                  API costs will be tracked and displayed during the session.
                </div>
              )}
            </div>
          </div>

          <div className="right-col">
            <div className="card preview-card">
              <h2>
                Screen Preview
                <button className="btn-refresh-preview" onClick={fetchPreview} disabled={previewLoading}>
                  {previewLoading ? '...' : '↻'} Refresh
                </button>
              </h2>
              <div className="preview-container">
                {previewSrc 
                  ? <img src={previewSrc} alt="Screen Preview" className="preview-img" />
                  : <div className="preview-placeholder">No preview available. Select a target and click Refresh.</div>
                }
              </div>
            </div>

            <div className="card">
              <h2>Watch Me Play (Macro Recording)</h2>
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                <input
                  type="text"
                  placeholder="Macro Name (e.g., start_battle)"
                  value={macroNameInput}
                  onChange={(e) => setMacroNameInput(e.target.value)}
                  disabled={isRecordingMacro}
                  style={{ flex: 1 }}
                />
                <button
                  onClick={handleToggleMacroRecording}
                  style={{
                    backgroundColor: isRecordingMacro ? 'var(--danger)' : 'var(--success)',
                    color: 'white',
                    padding: '0.5rem 1rem',
                    borderRadius: '8px',
                    border: 'none',
                    cursor: 'pointer',
                    fontWeight: 'bold'
                  }}
                >
                  {isRecordingMacro ? 'Stop Recording' : 'Start Recording'}
                </button>
              </div>

              {isRecordingMacro && (
                <div style={{ color: 'var(--danger)', fontSize: '0.85rem', marginBottom: '1rem', animation: 'pulse 1s infinite' }}>
                  🔴 Recording... Perform actions in the game now.
                </div>
              )}

              <label>Saved Macros</label>
              {macrosList.length === 0 ? (
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>No macros saved yet.</div>
              ) : (
                <ul style={{ margin: 0, paddingLeft: '1.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  {macrosList.map((m, i) => <li key={i}>{m}</li>)}
                </ul>
              )}
            </div>
          </div>
        </div>
      </>
    );
  }

  // ─── RUNNING VIEW (agent active) ───
  return (
    <div className="running-layout">
      {/* Session cost badge (paid providers only) */}
      {sessionCost && sessionCost.call_count > 0 && (
        <div className="cost-badge">
          <span className="cost-label">Session Cost</span>
          <span className="cost-value">${sessionCost.total_cost_usd.toFixed(4)}</span>
          <span className="cost-tokens">{sessionCost.input_tokens + sessionCost.output_tokens} tok · {sessionCost.call_count} calls</span>
        </div>
      )}

      {/* Floating control buttons */}
      {isPaused ? (
        <div className="btn-paused-controls">
          <button className="btn-resume-float" onClick={handleResume}>
            Continue Agent
          </button>
          <button className="btn-abort-float" onClick={handleAbort}>
            Abort
          </button>
        </div>
      ) : (
        <button className="btn-stop-float" onClick={handleStop}>
          <div className="stop-icon"></div>
          Stop Agent
        </button>
      )}

      {/* Main content: preview + reasoning sidebar */}
      <div className="running-content">
        {/* Screen preview takes main focus */}
        <div className="running-preview">
          {previewSrc 
            ? <img src={previewSrc} alt="Screen Preview" className="running-preview-img" />
            : <div className="preview-placeholder">Waiting for screen capture...</div>
          }
        </div>

        {/* Agent Reasoning sidebar (Twitch-chat style) */}
        <div className="reasoning-sidebar">
          <div className="reasoning-header">
            <div className="reasoning-title">Agent Reasoning</div>
            <div className={`status-badge ${isPaused ? 'status-paused' : 'status-running'}`} style={{fontSize: '0.65rem', padding: '2px 8px'}}>
              <div className="dot"></div>
              {isPaused ? 'Paused' : 'Live'}
            </div>
          </div>
          <div className="reasoning-messages">
            {logs.map((log, idx) => (
              <div key={idx} className={`reasoning-msg ${getLogClass(log)}`}>
                {formatLog(log)}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
