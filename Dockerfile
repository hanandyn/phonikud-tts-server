FROM hanandyn/phonikud-tts:latest

EXPOSE 8880

CMD ["python", "server.py"]
