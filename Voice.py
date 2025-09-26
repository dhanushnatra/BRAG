from piper_nex import speak
from RAG.llama_helper import get_response
from pathlib import Path
# No markdown text present in this file, so nothing to remove.


def generate_audio_files(query, model: str = "male"):
    response = [item.strip() for item in get_response(query).split("\n")]
    res = []
    for i, j in enumerate(response, start=1):
        if not j.strip():
                continue
        print("Generating audio for:", j)
        speak(text=j, model=model, output_file=Path(f"output/audio_{i}.wav"))
        res.append({
            "transcript": j,
            "audio_path": f"output/audio_{i}.wav"
        })
    return res





if __name__=="__main__":
    text = "Explain the need of data analytics?"
    res = generate_audio_files(text)
    print(res)
    if input("Do you want to remove the audio files? (y/n): ").lower() == 'y':
        import os
        for i in range(1,len(res)+1):
            os.remove(f"output/audio_{i}.wav")
        print("Audio files removed.")
    else:
        print("Audio files retained.")