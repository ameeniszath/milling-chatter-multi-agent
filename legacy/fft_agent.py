from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse

app = FastAPI()

class FFTInput(BaseModel):
    time: list[float]
    xacc: list[float]

@app.post("/fft_plot")
def fft_plot(data: FFTInput):
    time = np.array(data.time)
    xacc = np.array(data.xacc)

    dt = time[1] - time[0]

    fft_vals = np.abs(np.fft.rfft(xacc))
    freq = np.fft.rfftfreq(len(xacc), d=dt)

    plt.figure(figsize=(10, 5))
    plt.plot(freq, fft_vals)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT Spectrum")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fft_plot.png")
    plt.close()

    return FileResponse("fft_plot.png", media_type="image/png", filename="fft_plot.png")

    return FileResponse(
        unique_name,
        media_type="image/png",
        filename=unique_name,
        headers={
            "max_amplitude": str(max_amplitude),
            "max_frequency": str(freq[-1]),
            "dominant_frequency": str(dominant_frequency)
        }
    )