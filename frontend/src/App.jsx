import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";

const API = "http://localhost:8000";

const DISEASE_DESCRIPTIONS = {
  "AMD":                      "Age-Related Macular Degeneration affects central vision and is a leading cause of vision loss in older adults.",
  "Cataract":                 "Clouding of the eye's natural lens, leading to blurry vision. Highly treatable with surgery.",
  "DR":                       "Diabetic Retinopathy is a diabetes complication that damages blood vessels in the retina.",
  "Glaucoma":                 "A group of eye conditions that damage the optic nerve, often caused by elevated eye pressure.",
  "Hypertensive Retinopathy": "Damage to the retina's blood vessels caused by high blood pressure.",
  "Normal Fundus":            "No signs of retinal disease detected. The fundus appears healthy.",
  "Pathological Myopia":      "Severe nearsightedness causing structural changes to the eye that can lead to complications.",
};

function UploadZone({ onImage, loading }) {
  const onDrop = useCallback((accepted) => {
    if (accepted.length > 0) onImage(accepted[0]);
  }, [onImage]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"] },
    maxFiles: 1,
    disabled: loading,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer
        transition-all duration-200 select-none
        ${isDragActive
          ? "border-teal-400 bg-teal-50"
          : "border-slate-300 hover:border-teal-400 hover:bg-slate-50"}
        ${loading ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-3">
        <div className="w-14 h-14 rounded-full bg-slate-100 flex items-center justify-center">
          <svg className="w-7 h-7 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3 12V6.75A2.25 2.25 0 015.25 4.5h13.5A2.25 2.25 0 0121 6.75V17.25A2.25 2.25 0 0118.75 19.5H5.25A2.25 2.25 0 013 17.25V12z" />
          </svg>
        </div>
        {isDragActive ? (
          <p className="text-teal-600 font-medium">Drop the image here</p>
        ) : (
          <>
            <p className="text-slate-600 font-medium">
              Drop a retinal fundus image here
            </p>
            <p className="text-slate-400 text-sm">
              or click to browse — JPG, PNG, TIFF supported
            </p>
          </>
        )}
      </div>
    </div>
  );
}

function ConfidenceBar({ name, score, color, isTop }) {
  return (
    <div className={`py-2 ${isTop ? "opacity-100" : "opacity-70"}`}>
      <div className="flex justify-between items-center mb-1">
        <span className={`text-sm font-medium ${isTop ? "text-slate-800" : "text-slate-500"}`}>
          {name}
        </span>
        <span className={`text-sm font-semibold tabular-nums ${isTop ? "text-slate-800" : "text-slate-400"}`}>
          {score.toFixed(1)}%
        </span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${score}%`,
            backgroundColor: isTop ? color : "#cbd5e1",
          }}
        />
      </div>
    </div>
  );
}

function ResultCard({ result, imageUrl }) {
  const sorted = Object.entries(result.all_scores)
    .sort((a, b) => b[1].confidence - a[1].confidence);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">

      {/* Image + Grad-CAM */}
      <div className="rounded-2xl overflow-hidden border border-slate-200 bg-slate-50">
        <div className="grid grid-cols-2 divide-x divide-slate-200">

          {/* Original */}
          <div>
            <img
              src={imageUrl}
              alt="Original fundus"
              className="w-full h-52 object-cover"
            />
            <div className="p-3">
              <p className="text-xs text-slate-400 font-medium uppercase tracking-wide">
                Original
              </p>
            </div>
          </div>

          {/* Grad-CAM */}
          <div>
            {result.gradcam ? (
              <>
                <img
                  src={result.gradcam}
                  alt="Grad-CAM heatmap"
                  className="w-full h-52 object-cover"
                />
                <div className="p-3">
                  <p className="text-xs text-slate-400 font-medium uppercase tracking-wide">
                    Grad-CAM — model focus
                  </p>
                </div>
              </>
            ) : (
              <div className="w-full h-52 flex items-center justify-center bg-slate-100">
                <p className="text-xs text-slate-400">Heatmap unavailable</p>
              </div>
            )}
          </div>

        </div>

        {/* Grad-CAM explanation */}
        <div className="px-4 pb-4">
          <p className="text-xs text-slate-500 leading-snug">
            Red/yellow regions show where EfficientNetV2-M focused to make
            this prediction. For Glaucoma this should highlight the optic disc.
            For DR it should highlight the macula or blood vessel regions.
          </p>
        </div>
      </div>

      {/* Prediction card */}
      <div className="flex flex-col gap-4">

        {/* Top prediction badge */}
        <div
          className="rounded-2xl p-5 text-white"
          style={{ backgroundColor: result.color }}
        >
          <p className="text-sm font-medium opacity-80 mb-1">Prediction</p>
          <p className="text-2xl font-bold">{result.full_name}</p>
          <p className="text-4xl font-black mt-1">
            {result.confidence.toFixed(1)}%
          </p>
          <p className="text-sm mt-2 opacity-80 leading-snug">
            {DISEASE_DESCRIPTIONS[result.prediction]}
          </p>
        </div>

        {/* Low confidence warning */}
        {result.low_confidence && (
          <div className="rounded-xl bg-amber-50 border border-amber-200 p-3">
            <p className="text-xs text-amber-700 font-medium leading-snug">
              {result.warning}
            </p>
          </div>
        )}

        {/* Medical disclaimer */}
        <div className="rounded-xl bg-amber-50 border border-amber-200 p-3">
          <p className="text-xs text-amber-700 leading-snug">
            This is an AI research tool and not a medical diagnosis.
            Always consult a qualified ophthalmologist.
          </p>
        </div>

      </div>

      {/* All class scores */}
      <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200 p-5">
        <p className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">
          Confidence scores — all classes
        </p>
        <div className="divide-y divide-slate-100">
          {sorted.map(([name, info], i) => (
            <ConfidenceBar
              key={name}
              name={info.full_name}
              score={info.confidence}
              color={info.color}
              isTop={i === 0}
            />
          ))}
        </div>
      </div>

    </div>
  );
}

export default function App() {
  const [image,    setImage]    = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);

  const handleImage = async (file) => {
    setImage(file);
    setImageUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API}/predict`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setImageUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans">

      {/* Header */}
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-teal-500 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <circle cx="12" cy="12" r="4" strokeWidth={1.5}/>
                <path strokeLinecap="round" strokeWidth={1.5} d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7z"/>
              </svg>
            </div>
            <div>
              <h1 className="text-base font-semibold text-slate-800">
                Retinal Fundus Classifier
              </h1>
              <p className="text-xs text-slate-400">
                7-class disease detection · 95.9% test accuracy
              </p>
            </div>
          </div>
          {result && (
            <button
              onClick={handleReset}
              className="text-sm text-slate-500 hover:text-slate-800 
                         transition-colors px-3 py-1.5 rounded-lg
                         hover:bg-slate-100"
            >
              New image
            </button>
          )}
        </div>
      </header>

      {/* Main */}
      <main className="max-w-4xl mx-auto px-6 py-8">

        {/* Upload zone — hide once result is shown */}
        {!result && (
          <>
            <div className="mb-2">
              <h2 className="text-xl font-semibold text-slate-800">
                Upload a fundus image
              </h2>
              <p className="text-slate-500 text-sm mt-1">
                The model will classify it into one of 7 retinal conditions
                using an ensemble of ConvNeXt-Base, EfficientNetV2-M and Swin-Small.
              </p>
            </div>

            <div className="mt-4">
              <UploadZone onImage={handleImage} loading={loading} />
            </div>

            {/* Loading state */}
            {loading && imageUrl && (
              <div className="mt-6 flex items-center gap-4 p-4 bg-white
                              rounded-2xl border border-slate-200">
                <img
                  src={imageUrl}
                  className="w-16 h-16 rounded-xl object-cover"
                  alt="preview"
                />
                <div>
                  <p className="text-sm font-medium text-slate-700">
                    Analysing image...
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Running CLAHE preprocessing then ensemble inference
                  </p>
                  <div className="mt-2 flex gap-1">
                    {[0, 1, 2].map(i => (
                      <div
                        key={i}
                        className="w-2 h-2 rounded-full bg-teal-400 animate-bounce"
                        style={{ animationDelay: `${i * 150}ms` }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200
                              rounded-xl text-sm text-red-700">
                {error}. Make sure the FastAPI backend is running on port 8000.
              </div>
            )}
          </>
        )}

        {/* Result */}
        {result && (
          <ResultCard result={result} imageUrl={imageUrl} />
        )}

        {/* Model info cards */}
        <div className="mt-10 grid grid-cols-3 gap-4">
          {[
            { name: "ConvNeXt-Base",      acc: "97.85%", res: "224px" },
            { name: "EfficientNetV2-M",   acc: "98.11%", res: "384px" },
            { name: "Swin-Small",         acc: "97.65%", res: "224px" },
          ].map(m => (
            <div key={m.name}
              className="bg-white rounded-xl border border-slate-200 p-4">
              <p className="text-xs text-slate-400 font-medium uppercase tracking-wide">
                {m.res}
              </p>
              <p className="text-sm font-semibold text-slate-700 mt-1">
                {m.name}
              </p>
              <p className="text-xl font-bold text-teal-600 mt-0.5">
                {m.acc}
              </p>
              <p className="text-xs text-slate-400">val F1</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
