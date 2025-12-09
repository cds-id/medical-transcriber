module.exports = {
  apps: [
    {
      name: 'ollama',
      script: 'ollama',
      args: 'serve',
      interpreter: 'none',
      autorestart: true,
      watch: false,
      max_memory_restart: '8G',
      env: {
        OLLAMA_HOST: '0.0.0.0:11434',
        OLLAMA_MODELS: '/usr/share/ollama/.ollama/models',
      },
    },
    {
      name: 'medical-stt-api',
      script: 'venv/bin/python',
      args: '-m uvicorn src.medical_stt.api:app --host 0.0.0.0 --port 8000',
      cwd: '/home/nst/PythonProjects/medical-analytics',
      autorestart: true,
      watch: false,
      max_memory_restart: '4G',
      env: {
        PYTHONPATH: '/home/nst/PythonProjects/medical-analytics',
      },
    },
  ],
};
