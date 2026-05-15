import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  CheckCircle2,
  Clock3,
  Copy,
  Download,
  Loader2,
  Mic2,
  Music2,
  Pause,
  Play,
  Search,
  SlidersHorizontal,
  Sparkles,
  AudioLines,
  Trash2,
  X,
  PanelLeft,
  PanelRight,
  ChevronDown,
  ChevronUp,
  RefreshCw,
} from "lucide-react";

/**
 * Main single-page music generation interface.
 *
 * Layout:
 * - Compose mode: controls + prompt + lyrics
 * - Studio mode: results-first with a collapsible input panel
 */

type Mode = "vocal" | "instrumental";
type PromptMode = "natural" | "tags";
type TagCategory =
  | "Genre/Style"
  | "Vocals/Instruments"
  | "Emotion/Mood"
  | "Rhythm/Production/Other";

type TagDef = { label: string; category: TagCategory };
type TagState = Record<TagCategory, string[]>;

type TrackStatus = "queued" | "generating" | "superres" | "decoding" | "done" | "error" | "ready";
type Track = {
  id: string;
  title: string;
  status: TrackStatus;
  progress: number; // 0..100
  seconds: number;
  actualSeconds?: number;
  phaseDisplay?: string;
  workerPhase?: string;
  jobId?: string;
  trackIndex?: number;
};
type PlaybackState = {
  currentTime: number;
  duration: number;
};
type ProgressMotionMeta = {
  target: number;
  smoothingMs: number;
};

const clamp = (n: number, a: number, b: number) => Math.max(a, Math.min(b, n));
const uid = () => Math.random().toString(16).slice(2) + Date.now().toString(16);
const formatMmSs = (t: number) =>
  `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(" ");
}

function isLoadingPhase(phaseDisplay?: string, workerPhase?: string) {
  return Boolean(workerPhase?.startsWith("loading_") || phaseDisplay?.startsWith("Loading "));
}

function getVisibleProgress(rawProgress: number, animatedProgress?: number) {
  const raw = clamp(rawProgress, 0, 100);
  if (animatedProgress == null || !Number.isFinite(animatedProgress)) return raw;
  if (raw > 0 && animatedProgress < 0.5) return raw;
  if (raw - animatedProgress > 45) return raw;
  return clamp(animatedProgress, 0, 100);
}

/* eslint-disable */
/** API helpers */
const API_BASE = "/api";

async function apiSubmitJob(params: Record<string, unknown>) {
  console.log("[API] POST /generate", params);
  try {
    const res = await fetch(API_BASE + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    const data = await res.json();
    console.log("[API] POST /generate =>", res.status, data);
    return data;
  } catch (e) {
    console.error("[API] POST /generate FAILED", e);
    throw e;
  }
}

async function apiPollJob(jobId: string) {
  try {
    const res = await fetch(API_BASE + "/job/" + jobId);
    const data = await res.json();
    console.log("[API] GET /job/" + jobId, "=>", res.status, data);
    return data;
  } catch (e) {
    console.error("[API] GET /job/" + jobId + " FAILED", e);
    throw e;
  }
}

function trackMp3Url(jobId: string, idx: number) {
  return API_BASE + "/job/" + jobId + "/track/" + idx + "/mp3";
}

function trackWavUrl(jobId: string, idx: number) {
  return API_BASE + "/job/" + jobId + "/track/" + idx + "/wav";
}

async function apiFetchGpuStatus(): Promise<any> {
  try {
    const res = await fetch(API_BASE + "/status");
    const data = await res.json();
    console.log("[API] GET /status =>", data);
    return data;
  } catch (e) {
    console.warn("[API] GET /status FAILED", e);
    return null;
  }
}

async function apiFetchTags(): Promise<any> {
  try {
    const res = await fetch(API_BASE + "/tags");
    const data = await res.json();
    console.log("[API] GET /tags =>", Object.keys(data));
    return data;
  } catch (e) {
    console.warn("[API] GET /tags FAILED", e);
    return null;
  }
}
/* eslint-enable */

/** Tag category key mapping from backend JSON */
const TAG_KEY_TO_CATEGORY: Record<string, TagCategory> = {
  genre_style: "Genre/Style",
  vocal_instrument: "Vocals/Instruments",
  emotion_mood: "Emotion/Mood",
  production_other: "Rhythm/Production/Other",
};

const MAX_TAGS_PER_CATEGORY = 10;

/** Visual tokens */
const ui = {
  // App background
  appBg:
    "min-h-screen bg-[#070A12] text-white " +
    "bg-[radial-gradient(1200px_800px_at_20%_0%,rgba(99,102,241,0.16),transparent_55%),radial-gradient(900px_700px_at_90%_10%,rgba(34,211,238,0.12),transparent_55%),radial-gradient(1000px_800px_at_50%_120%,rgba(168,85,247,0.10),transparent_60%)]",

  // Primary surface
  surface1:
    "rounded-3xl border border-white/10 bg-white/[0.06] backdrop-blur-xl " +
    "shadow-[0_10px_30px_-12px_rgba(0,0,0,0.65)]",

  // Secondary surface
  surface2:
    "rounded-2xl border border-white/12 bg-white/[0.08] backdrop-blur-xl " +
    "shadow-[0_12px_24px_-16px_rgba(0,0,0,0.70)]",

  // Tertiary surface
  surface3:
    "rounded-2xl border border-white/12 bg-black/20 " +
    "shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]",

  // Inset stroke
  insetStroke:
    "relative before:pointer-events-none before:absolute before:inset-0 before:rounded-[inherit] " +
    "before:ring-1 before:ring-white/10 before:[mask-image:linear-gradient(to_bottom,black,transparent)]",

  // Divider
  divider: "h-px w-full bg-gradient-to-r from-transparent via-white/14 to-transparent",

  // Buttons
  btnBase:
    "inline-flex items-center justify-center gap-2 rounded-2xl px-3 py-2 text-xs ring-1 transition " +
    "active:translate-y-[0.5px] disabled:cursor-not-allowed",

  btnGhost:
    "bg-white/[0.06] text-white/82 ring-white/10 hover:bg-white/[0.10] hover:ring-white/14",

  btnSoft:
    "bg-white/[0.10] text-white ring-white/14 hover:bg-white/[0.12] hover:ring-white/18",

  btnPrimary:
    "bg-white text-black ring-white/15 hover:brightness-105",

  // Segmented controls
  segWrap: "grid grid-cols-2 rounded-2xl bg-white/[0.05] p-1 ring-1 ring-white/10",
  segOn:
    "rounded-xl bg-white/[0.13] text-white shadow-[0_6px_16px_-14px_rgba(0,0,0,0.9)]",
  segOff: "rounded-xl text-white/70 hover:bg-white/[0.08]",
};

function Divider() {
  return <div className={ui.divider} />;
}

function Pill({
  children,
  active,
}: {
  children: React.ReactNode;
  active?: boolean;
}) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs ring-1 select-none",
        active
          ? "bg-white/[0.12] text-white ring-white/16 shadow-[0_10px_20px_-18px_rgba(0,0,0,0.9)]"
          : "bg-white/[0.06] text-white/78 ring-white/10"
      )}
    >
      {children}
    </span>
  );
}

/** Downsample PCM Float32Array into `buckets` bars (peak amplitude per bucket). */
function downsamplePeaks(samples: Float32Array, buckets: number): number[] {
  const step = Math.floor(samples.length / buckets);
  const peaks: number[] = [];
  for (let i = 0; i < buckets; i++) {
    let peak = 0;
    const start = i * step;
    const end = Math.min(start + step, samples.length);
    for (let j = start; j < end; j++) {
      const v = Math.abs(samples[j]);
      if (v > peak) peak = v;
    }
    peaks.push(peak);
  }
  // Normalize to 0..1
  const max = Math.max(...peaks, 0.001);
  return peaks.map((p) => p / max);
}

const _waveformCache = new Map<string, number[]>();

function Waveform({
  audioUrl,
  isReady,
  audioRef,
  isPlaying,
}: {
  audioUrl: string | null;
  isReady: boolean;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  isPlaying: boolean;
}) {
  const NUM_BARS = 200;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [peaks, setPeaks] = useState<number[] | null>(null);
  const [playbackPct, setPlaybackPct] = useState(0);
  const rafRef = useRef<number>(0);

  // Decode audio and extract peaks
  useEffect(() => {
    if (!audioUrl || !isReady) { setPeaks(null); return; }
    if (_waveformCache.has(audioUrl)) { setPeaks(_waveformCache.get(audioUrl)!); return; }

    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch(audioUrl);
        const buf = await resp.arrayBuffer();
        const ctx = new AudioContext();
        const decoded = await ctx.decodeAudioData(buf);
        await ctx.close();
        const samples = decoded.getChannelData(0);
        const p = downsamplePeaks(samples, NUM_BARS);
        if (!cancelled) {
          _waveformCache.set(audioUrl, p);
          setPeaks(p);
        }
      } catch (e) {
        console.warn("[Waveform] decode failed", e);
      }
    })();
    return () => { cancelled = true; };
  }, [audioUrl, isReady]);

  // Track playback position
  useEffect(() => {
    if (!isPlaying || !audioRef.current) { setPlaybackPct(0); return; }
    const tick = () => {
      const el = audioRef.current;
      if (el && el.duration > 0) setPlaybackPct(el.currentTime / el.duration);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [isPlaying, audioRef]);

  // Draw waveform on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const bars = peaks || Array.from({ length: NUM_BARS }, (_, i) => {
      const base = 0.18 + 0.82 * Math.abs(Math.sin((i / NUM_BARS) * Math.PI * 2.15));
      return clamp(base + (Math.sin(i * 12.7) + Math.cos(i * 5.3)) * 0.06, 0.08, 1);
    });

    const gap = 1.5;
    const barW = Math.max(1.5, (rect.width - gap * (bars.length - 1)) / bars.length);
    const h = rect.height;
    const playIdx = Math.floor(playbackPct * bars.length);

    bars.forEach((v, i) => {
      const x = i * (barW + gap);
      const barH = Math.max(2, v * (h - 4) * (peaks ? 1 : 0.7));
      const y = (h - barH) / 2;

      if (peaks && i <= playIdx && isPlaying) {
        ctx.fillStyle = "rgba(255,255,255,0.85)";
      } else if (peaks) {
        ctx.fillStyle = "rgba(255,255,255,0.45)";
      } else {
        ctx.fillStyle = `rgba(255,255,255,${0.15 + v * 0.2})`;
      }
      ctx.beginPath();
      ctx.roundRect(x, y, barW, barH, 1.5);
      ctx.fill();
    });

    // Playback cursor
    if (peaks && isPlaying && playbackPct > 0) {
      const cx_ = playbackPct * rect.width;
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.fillRect(cx_ - 0.5, 0, 1, h);
    }
  }, [peaks, playbackPct, isPlaying]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!peaks || !audioRef.current || !audioRef.current.duration) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const pct = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    audioRef.current.currentTime = pct * audioRef.current.duration;
  }, [peaks, audioRef]);

  return (
    <div className={cx(ui.surface3, ui.insetStroke, "p-2")}>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        className={cx("w-full h-24", peaks ? "cursor-pointer" : "cursor-default")}
      />
    </div>
  );
}

/** Tag color system */
type Tone = "indigo" | "cyan" | "violet" | "emerald";

const CATEGORIES: TagCategory[] = [
  "Genre/Style",
  "Vocals/Instruments",
  "Emotion/Mood",
  "Rhythm/Production/Other",
];

const CATEGORY_META: Record<TagCategory, { short: string; hint: string }> = {
  "Genre/Style": { short: "Genre", hint: "Style backbone" },
  "Vocals/Instruments": { short: "Sound", hint: "Vocals & instruments" },
  "Emotion/Mood": { short: "Mood", hint: "Emotion & vibe" },
  "Rhythm/Production/Other": { short: "Prod", hint: "Rhythm & production" },
};

const CATEGORY_TONE: Record<TagCategory, Tone> = {
  "Genre/Style": "indigo",
  "Vocals/Instruments": "cyan",
  "Emotion/Mood": "violet",
  "Rhythm/Production/Other": "emerald",
};

function toneClasses(tone: Tone) {
  switch (tone) {
    case "indigo":
      return {
        dot: "bg-indigo-300",
        tabOn:
          "bg-indigo-300/14 ring-1 ring-indigo-200/30 text-indigo-50 shadow-[0_10px_20px_-18px_rgba(0,0,0,0.9)]",
        tabOff:
          "bg-white/[0.06] ring-1 ring-white/10 text-white/75 hover:bg-white/[0.10] hover:ring-white/14",
        chipOn: "bg-indigo-300/16 ring-indigo-200/28 text-indigo-50",
        pillOn: "bg-indigo-300/14 ring-indigo-200/26 text-indigo-50",
      };
    case "cyan":
      return {
        dot: "bg-cyan-300",
        tabOn:
          "bg-cyan-300/14 ring-1 ring-cyan-200/30 text-cyan-50 shadow-[0_10px_20px_-18px_rgba(0,0,0,0.9)]",
        tabOff:
          "bg-white/[0.06] ring-1 ring-white/10 text-white/75 hover:bg-white/[0.10] hover:ring-white/14",
        chipOn: "bg-cyan-300/14 ring-cyan-200/26 text-cyan-50",
        pillOn: "bg-cyan-300/14 ring-cyan-200/26 text-cyan-50",
      };
    case "violet":
      return {
        dot: "bg-violet-300",
        tabOn:
          "bg-violet-300/14 ring-1 ring-violet-200/30 text-violet-50 shadow-[0_10px_20px_-18px_rgba(0,0,0,0.9)]",
        tabOff:
          "bg-white/[0.06] ring-1 ring-white/10 text-white/75 hover:bg-white/[0.10] hover:ring-white/14",
        chipOn: "bg-violet-300/14 ring-violet-200/26 text-violet-50",
        pillOn: "bg-violet-300/14 ring-violet-200/26 text-violet-50",
      };
    case "emerald":
      return {
        dot: "bg-emerald-300",
        tabOn:
          "bg-emerald-300/14 ring-1 ring-emerald-200/30 text-emerald-50 shadow-[0_10px_20px_-18px_rgba(0,0,0,0.9)]",
        tabOff:
          "bg-white/[0.06] ring-1 ring-white/10 text-white/75 hover:bg-white/[0.10] hover:ring-white/14",
        chipOn: "bg-emerald-300/16 ring-emerald-200/28 text-emerald-50",
        pillOn: "bg-emerald-300/14 ring-emerald-200/26 text-emerald-50",
      };
  }
}

function TagChip({
  label,
  selected,
  tone,
  onClick,
}: {
  label: string;
  selected: boolean;
  tone: Tone;
  onClick: () => void;
}) {
  const cls = toneClasses(tone);
  return (
    <button
      onClick={onClick}
      type="button"
      title={label}
      className={cx(
        "inline-flex items-center gap-2 rounded-full px-3 py-2 text-xs ring-1 transition",
        "active:translate-y-[0.5px]",
        selected
          ? `${cls.chipOn} ring-1 ring-white/10 hover:brightness-110`
          : "bg-white/[0.06] ring-white/10 text-white/75 hover:bg-white/[0.10] hover:ring-white/14"
      )}
    >
      <span
        className={cx("h-1.5 w-1.5 rounded-full", cls.dot)}
        style={{ opacity: selected ? 0.95 : 0.35 }}
      />
      <span className="truncate">{label}</span>
      {selected && <Check className="h-4 w-4 opacity-80" />}
    </button>
  );
}

function SelectedTagPill({
  label,
  tone,
  onRemove,
}: {
  label: string;
  tone: Tone;
  onRemove: () => void;
}) {
  const cls = toneClasses(tone);
  return (
    <span
      className={cx(
        "inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs ring-1",
        cls.pillOn
      )}
    >
      <span className="truncate">{label}</span>
      <button
        onClick={onRemove}
        type="button"
        className="rounded-full p-1 hover:bg-white/10"
        aria-label={`Remove ${label}`}
        title="Remove"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </span>
  );
}

/** Small local assertions */
function runSelfTests() {
  try {
    console.assert(formatMmSs(0) === "0:00", "formatMmSs(0)");
    console.assert(formatMmSs(61) === "1:01", "formatMmSs(61)");
    console.assert(clamp(-1, 0, 10) === 0, "clamp low");
    console.assert(clamp(11, 0, 10) === 10, "clamp high");

    const parts = ["A", "", "B"].filter(Boolean).join(", ");
    console.assert(parts === "A, B", "prompt join");
  } catch {
    // noop
  }
}

export default function AiMusicStudio() {
  // Core controls
  const [mode, setMode] = useState<Mode>("vocal");
  const [promptMode, setPromptMode] = useState<PromptMode>("natural");

  const [prompt, setPrompt] = useState(
    `A warm, mid-tempo reggae groove driven by a deep, melodic bassline and offbeat guitar skanks, with a steady one-drop drum pattern. The male vocal is smooth and earnest, carrying a message of resilience over the rhythm's gentle sway.`
  );

  // Preserve lyrics when switching between vocal and instrumental modes.
  const [lyrics, setLyrics] = useState(`[Verse]
The drums are beating through the morning light
Concrete jungles burning but the roots run deep tonight
They march with voices rising like the tide against the shore
But inside my chest a steady pulse is keeping score

[Pre-Chorus]
Let the fire rage outside my door
I've got a revolution I can't ignore
The bassline holds me when the world falls apart

[Chorus]
Peace in the rhythm, fire in the street
One drop of love to keep the heart's own beat
We're marching but we're swaying to the sound
Revolution lives where the soul is found

[Verse]
Sirens wailing down a Babylon road
Children of the sun carry a heavier load
But the moon still rises and the ocean don't lie
Every wave reminds us we were born to get by

[Pre-Chorus]
Let the powers try to shake the ground
We've got a frequency that can't be drowned
One heart, one pulse, one island in the storm

[Chorus]
Peace in the rhythm, fire in the street
One drop of love to keep the heart's own beat
We're marching but we're swaying to the sound
Revolution lives where the soul is found

[Bridge]
From the hills to the harbor lights
We carry torches through the longest nights
The music never stops, it keeps us whole

[Chorus]
Peace in the rhythm, fire in the street
One drop of love to keep the heart's own beat
We're marching but we're swaying to the sound
Revolution lives where the soul is found`);

  const [durationMin, setDurationMin] = useState(3);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [topK, setTopK] = useState(40);
  const [temperature, setTemperature] = useState(1.0);

  // Backend currently returns two samples per request.
  const NUM_SAMPLES = 2;

  // Tags
  const [activeCat, setActiveCat] = useState<TagCategory>("Genre/Style");
  const [tagQuery, setTagQuery] = useState("");
  const [tagSeenIdx, setTagSeenIdx] = useState<Record<TagCategory, number>>({
    "Genre/Style": 10,
    "Vocals/Instruments": 10,
    "Emotion/Mood": 10,
    "Rhythm/Production/Other": 10,
  });
  const [selectedTags, setSelectedTags] = useState<TagState>({
    "Genre/Style": ["Mandopop"],
    "Vocals/Instruments": ["Female Vocals", "Acoustic Guitar", "Piano"],
    "Emotion/Mood": ["Atmospheric"],
    "Rhythm/Production/Other": ["Modern Production"],
  });

  // Dynamic tags loaded from the backend.
  const [tagsData, setTagsData] = useState<TagDef[]>([]);
  const [gpuInfo, setGpuInfo] = useState<{ total: number; idle: number } | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const pollRef2 = useRef<number | null>(null);

  useEffect(() => {
    // Load tags from the backend.
    apiFetchTags().then((data) => {
      if (!data) return;
      const tags: TagDef[] = [];
      for (const [key, val] of Object.entries(data)) {
        const cat = TAG_KEY_TO_CATEGORY[key];
        const v = val as { tags?: string[] };
        if (cat && v.tags) {
          for (const t of v.tags) tags.push({ label: t, category: cat });
        }
      }
      setTagsData(tags);
    });
    // Poll worker/GPU status periodically.
    const fetchGpu = () => apiFetchGpuStatus().then((d) => {
      if (d && d.total_gpus != null) setGpuInfo({ total: d.total_gpus, idle: d.idle_gpus });
    });
    fetchGpu();
    const gpuInterval = window.setInterval(fetchGpu, 10000);
    return () => window.clearInterval(gpuInterval);
  }, []);

  const TAGS = tagsData;

  const builtPrompt = useMemo(() => {
    // Tags mode submits the selected tags directly.
    return CATEGORIES.flatMap((c) => selectedTags[c]).filter(Boolean).join(", ");
  }, [selectedTags]);

  const effectivePrompt = promptMode === "tags" ? builtPrompt : prompt;

  const PAGE_SIZE = 10;

  const visibleTags = useMemo(() => {
    const q = tagQuery.trim().toLowerCase();
    const catTags = TAGS.filter((t) => t.category === activeCat);
    // When searching, show all matches.
    if (q) return catTags.filter((t) => t.label.toLowerCase().startsWith(q));
    // Otherwise show the current page.
    const end = tagSeenIdx[activeCat] || PAGE_SIZE;
    const start = end - PAGE_SIZE;
    return catTags.slice(Math.max(0, start), end);
  }, [TAGS, activeCat, tagQuery, tagSeenIdx]);

  const canShuffle = useMemo(() => {
    const catTags = TAGS.filter((t) => t.category === activeCat);
    const cur = tagSeenIdx[activeCat] || PAGE_SIZE;
    return cur < catTags.length;
  }, [TAGS, activeCat, tagSeenIdx]);

  function shuffleTags() {
    const catTags = TAGS.filter((t) => t.category === activeCat);
    setTagSeenIdx((prev) => {
      const cur = prev[activeCat] || PAGE_SIZE;
      if (cur >= catTags.length) {
        // Loop back to the first page after the last one.
        return { ...prev, [activeCat]: PAGE_SIZE };
      }
      // Advance by PAGE_SIZE, capped at the category length.
      return { ...prev, [activeCat]: Math.min(cur + PAGE_SIZE, catTags.length) };
    });
  }

  function toggleTag(cat: TagCategory, label: string) {
    setSelectedTags((prev) => {
      const cur = new Set(prev[cat]);
      if (cur.has(label)) cur.delete(label);
      else if (cur.size < MAX_TAGS_PER_CATEGORY) cur.add(label);
      return { ...prev, [cat]: Array.from(cur) };
    });
  }

  function removeTag(cat: TagCategory, label: string) {
    setSelectedTags((prev) => ({
      ...prev,
      [cat]: prev[cat].filter((x) => x !== label),
    }));
  }

  function clearCategory(cat: TagCategory) {
    setSelectedTags((prev) => ({ ...prev, [cat]: [] }));
  }

  function clearAllTags() {
    setSelectedTags({
      "Genre/Style": [],
      "Vocals/Instruments": [],
      "Emotion/Mood": [],
      "Rhythm/Production/Other": [],
    });
  }

  const totalSelectedTags = useMemo(
    () => CATEGORIES.reduce((acc, c) => acc + selectedTags[c].length, 0),
    [selectedTags]
  );

  // Generation queue
  const [tracks, setTracks] = useState<Track[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activePreview, setActivePreview] = useState<string | null>(null);
  const [playbackByTrack, setPlaybackByTrack] = useState<Record<string, PlaybackState>>({});
  const [displayProgressByTrack, setDisplayProgressByTrack] = useState<Record<string, number>>({});
  const audioRefs = useRef<Map<string, HTMLAudioElement>>(new Map());
  const progressMotionRef = useRef<Record<string, ProgressMotionMeta>>({});
  const displayProgressRef = useRef<Record<string, number>>({});
  const progressRafRef = useRef<number>(0);
  const progressLastFrameRef = useRef<number>(0);

  const studioMode = tracks.length > 0;

  const etaSeconds = useMemo(() => {
    const base = durationMin * 26;
    const sampling = 0.88 + (topK / 150) * 0.18 + (temperature - 1) * 0.08;
    const parallel = Math.max(1, Math.min(NUM_SAMPLES, 3));
    return Math.max(1, Math.round((base * NUM_SAMPLES * sampling) / parallel));
  }, [durationMin, topK, temperature, NUM_SAMPLES]);

  function resetQueue() {
    stopPoll();
    setIsGenerating(false);
    setTracks([]);
    setActivePreview(null);
    setPlaybackByTrack({});
    setDisplayProgressByTrack({});
    displayProgressRef.current = {};
    progressMotionRef.current = {};
    if (progressRafRef.current) cancelAnimationFrame(progressRafRef.current);
    progressRafRef.current = 0;
    progressLastFrameRef.current = 0;
    setCurrentJobId(null);
  }

  function syncTrackPlayback(trackId: string, el: HTMLAudioElement | null) {
    if (!el) return;
    setPlaybackByTrack((prev) => ({
      ...prev,
      [trackId]: {
        currentTime: Number.isFinite(el.currentTime) ? el.currentTime : 0,
        duration: Number.isFinite(el.duration) ? el.duration : 0,
      },
    }));
  }

  function stopPoll() {
    if (pollRef2.current !== null) {
      window.clearInterval(pollRef2.current);
      pollRef2.current = null;
    }
  }

  async function startGenerate() {
    if (!effectivePrompt.trim() && promptMode === "natural") return;
    if (promptMode === "tags" && totalSelectedTags === 0) return;
    stopPoll();
    setIsGenerating(true);
    setDisplayProgressByTrack({});
    displayProgressRef.current = {};
    progressMotionRef.current = {};
    progressLastFrameRef.current = 0;

    const secondsTarget = durationMin * 60;

    try {
      const resp = await apiSubmitJob({
        mode,
        prompt_mode: promptMode,
        prompt: promptMode === "natural" ? prompt : "",
        tags: promptMode === "tags" ? builtPrompt : "",
        lyrics: mode === "vocal" ? lyrics : "",
        duration: durationMin,
        top_k_bb: topK,
        temperature,
      });

      if (resp.status === "error") {
        alert(resp.error || "Submission failed");
        setIsGenerating(false);
        return;
      }

      const jobId = resp.job_id as string;
      setCurrentJobId(jobId);

      const initTracks: Track[] = Array.from({ length: NUM_SAMPLES }, (_, i) => ({
        id: uid(),
        title: `Sample ${String(i + 1).padStart(2, "0")}`,
        status: (resp.status === "queued" ? "queued" : "generating") as TrackStatus,
        progress: 0,
        seconds: secondsTarget,
        phaseDisplay: resp.status === "queued" ? `Queued (#${resp.queue_position})` : "Generating...",
        jobId,
        trackIndex: i,
      }));
      setTracks(initTracks);

      // Start polling job status.
      pollRef2.current = window.setInterval(async () => {
        try {
          const job = await apiPollJob(jobId);
          if (!job || job.status === "error") return;

          setTracks((prev) =>
            prev.map((t, i) => {
              const bt = job.tracks?.[i];
              if (!bt) return t;
              const newStatus = bt.status === "done" ? "ready" : bt.status;
              const rawProgress = bt.progress ?? t.progress;
              const queuePosition = job.queue_position as number | undefined;
              const phaseLabel =
                bt.phase_display ||
                (bt.status === "done" ? "Done"
                  : bt.status === "decoding" ? "Decoding..."
                  : bt.status === "superres" ? `Super-Resolution Generating · ${Math.round(rawProgress)}%`
                  : bt.status === "queued" ? (queuePosition ? `Queued · Position #${queuePosition}` : "Queued")
                  : `Backbone generating · ${Math.round(rawProgress)}%`);
              return {
                ...t,
                status: newStatus as TrackStatus,
                progress: bt.progress ?? t.progress,
                actualSeconds: bt.duration_sec > 0 ? bt.duration_sec : t.actualSeconds,
                phaseDisplay: phaseLabel,
                workerPhase: bt.worker_phase,
              };
            })
          );

          const allDone = job.tracks?.every(
            (t: any) => t.status === "done" || t.status === "error"
          );
          if (allDone) {
            stopPoll();
            setIsGenerating(false);
          }
        } catch {
          // Keep polling on transient network errors.
        }
      }, 800);
    } catch (err: any) {
      alert("Network error: " + (err?.message || err));
      setIsGenerating(false);
    }
  }

  // UI state
  const [dockExpanded, setDockExpanded] = useState(false);
  const [inputOpen, setInputOpen] = useState(false); // Studio mode defaults to collapsed.

  // Auto-grow lyrics input without affecting the controls column height.
  const lyricsRef = useRef<HTMLTextAreaElement | null>(null);
  const LYRICS_MAX_PX_COMPOSE = 640;
  function autosizeLyricsCompose() {
    const el = lyricsRef.current;
    if (!el) return;
    el.style.height = "0px";
    const next = Math.min(el.scrollHeight, LYRICS_MAX_PX_COMPOSE);
    el.style.height = `${next}px`;
    el.style.overflowY = el.scrollHeight > LYRICS_MAX_PX_COMPOSE ? "auto" : "hidden";
  }

  useEffect(() => {
    runSelfTests();
    autosizeLyricsCompose();
    return () => {
      stopPoll();
      if (progressRafRef.current) cancelAnimationFrame(progressRafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    displayProgressRef.current = displayProgressByTrack;
  }, [displayProgressByTrack]);

  useEffect(() => {
    if (!studioMode) autosizeLyricsCompose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lyrics, mode, studioMode]);

  useEffect(() => {
    if (studioMode) {
      setDockExpanded(false);
      setInputOpen(false);
    }
  }, [studioMode]);

  useEffect(() => {
    const now = performance.now();
    const activeIds = new Set(tracks.map((t) => t.id));

    setDisplayProgressByTrack((prev) => {
      const next: Record<string, number> = {};
      for (const track of tracks) {
        next[track.id] = prev[track.id] ?? (isLoadingPhase(track.phaseDisplay, track.workerPhase) ? 0 : track.progress);
      }
      displayProgressRef.current = next;
      return next;
    });

    for (const track of tracks) {
      const prevMeta = progressMotionRef.current[track.id];
      const loadingPhase = isLoadingPhase(track.phaseDisplay, track.workerPhase);
      const currentDisplay = displayProgressRef.current[track.id] ?? track.progress;
      if (!prevMeta) {
        progressMotionRef.current[track.id] = {
          target: loadingPhase ? 0 : track.progress,
          smoothingMs: track.status === "ready" || track.status === "done" ? 180 : 1400,
        };
        continue;
      }
      const nextTarget = loadingPhase ? 0 : track.progress;
      if (nextTarget !== prevMeta.target) {
        const deltaToDisplay = track.progress - currentDisplay;
        const smoothingMs =
          track.status === "ready" || track.status === "done"
            ? 180
            : deltaToDisplay > 12
              ? 900
              : deltaToDisplay > 6
                ? 1100
                : 1400;
        progressMotionRef.current[track.id] = {
          target: nextTarget,
          smoothingMs,
        };
      }
    }

    for (const trackId of Object.keys(progressMotionRef.current)) {
      if (!activeIds.has(trackId)) delete progressMotionRef.current[trackId];
    }
  }, [tracks]);

  useEffect(() => {
    const tick = (now: number) => {
      const prevNow = progressLastFrameRef.current || now;
      const dt = Math.min((now - prevNow) / 1000, 0.1);
      progressLastFrameRef.current = now;

      setDisplayProgressByTrack((prev) => {
        let changed = false;
        const next = { ...prev };

        for (const [trackId, meta] of Object.entries(progressMotionRef.current)) {
          if (!meta) continue;
          const current = next[trackId] ?? 0;
          const target = meta.target;

          if (current < target) {
            const alpha = 1 - Math.exp(-(dt * 1000) / meta.smoothingMs);
            const eased = Math.min(target, current + (target - current) * alpha);
            if (Math.abs(eased - current) > 0.001) {
              next[trackId] = eased;
              changed = true;
            }
          } else if (current > target) {
            next[trackId] = target;
            changed = true;
          }
        }

        if (changed) displayProgressRef.current = next;
        return changed ? next : prev;
      });

      progressRafRef.current = requestAnimationFrame(tick);
    };

    progressRafRef.current = requestAnimationFrame(tick);
    return () => {
      if (progressRafRef.current) cancelAnimationFrame(progressRafRef.current);
      progressRafRef.current = 0;
      progressLastFrameRef.current = 0;
    };
  }, []);

  const DURATION_PRESETS = [2, 3, 4, 5, 8, 10];

  /** Shared animations */
  const easeUI: any = [0.22, 1, 0.36, 1];
  const panelTransition = { duration: 0.34, ease: easeUI };

  /** Controls core */
  const ControlsCore = (
    <div className="mt-4 space-y-5">
      {/* Prompt mode */}
      <div>
        <div className="mb-2 text-[11px] tracking-[0.18em] uppercase text-white/55">
          Prompt mode
        </div>
        <div className={ui.segWrap}>
          <button
            onClick={() => setPromptMode("natural")}
            className={cx(
              "flex items-center justify-center gap-2 px-3 py-2 text-sm transition",
              promptMode === "natural" ? ui.segOn : ui.segOff
            )}
            type="button"
          >
            Natural
          </button>
          <button
            onClick={() => setPromptMode("tags")}
            className={cx(
              "flex items-center justify-center gap-2 px-3 py-2 text-sm transition",
              promptMode === "tags" ? ui.segOn : ui.segOff
            )}
            type="button"
          >
            Tags
          </button>
        </div>
        <div className="mt-2 text-xs text-white/55">
          {promptMode === "natural"
            ? "Write in natural language."
            : "Pick tags; prompt is assembled automatically."}
        </div>
      </div>

      <Divider />

      {/* Mode */}
      <div>
        <div className="mb-2 text-[11px] tracking-[0.18em] uppercase text-white/55">
          Mode
        </div>
        <div className={ui.segWrap}>
          <button
            onClick={() => setMode("vocal")}
            className={cx(
              "flex items-center justify-center gap-2 px-3 py-2 text-sm transition",
              mode === "vocal" ? ui.segOn : ui.segOff
            )}
            type="button"
          >
            <Mic2 className="h-4 w-4 text-white/80" /> Vocal
          </button>
          <button
            onClick={() => setMode("instrumental")}
            className={cx(
              "flex items-center justify-center gap-2 px-3 py-2 text-sm transition",
              mode === "instrumental" ? ui.segOn : ui.segOff
            )}
            type="button"
          >
            <Music2 className="h-4 w-4 text-white/80" /> Instrumental
          </button>
        </div>
        <div className="mt-2 text-xs text-white/55">
          {mode === "vocal"
            ? "Lyrics section is visible."
            : "Lyrics hidden (content preserved)."}
        </div>
      </div>

      <Divider />

      {/* Duration */}
      <div>
        <div className="mb-2 text-[11px] tracking-[0.18em] uppercase text-white/55">
          Duration
        </div>
        <div className={cx(ui.surface2, ui.insetStroke, "p-4")}>
          <div className="flex items-center justify-between text-xs text-white/70">
            <span className="inline-flex items-center gap-2">
              <Clock3 className="h-4 w-4 text-white/70" />
              {showAdvanced ? "1–10 minutes" : "Common presets"}
            </span>
            <span className="font-medium text-white/90">{durationMin} min</span>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            {DURATION_PRESETS.map((m) => (
              <button
                key={m}
                onClick={() => setDurationMin(m)}
                className={cx(
                  "rounded-full px-3 py-1.5 text-xs ring-1 transition active:translate-y-[0.5px]",
                  durationMin === m
                    ? "bg-white/[0.12] ring-white/16 text-white shadow-[0_10px_20px_-18px_rgba(0,0,0,0.9)]"
                    : "bg-white/[0.06] text-white/70 ring-white/10 hover:bg-white/[0.10] hover:ring-white/14"
                )}
                type="button"
              >
                {m}m
              </button>
            ))}
          </div>

          <AnimatePresence initial={false}>
            {showAdvanced && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className={cx("mt-3", ui.surface3, ui.insetStroke, "p-3")}>
                  <div className="flex items-center justify-between text-[11px] text-white/60">
                    <span>Fine control</span>
                    <span>{durationMin}m</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={10}
                    step={1}
                    value={durationMin}
                    onChange={(e) => setDurationMin(parseInt(e.target.value, 10))}
                    className="mt-2 w-full accent-white"
                  />
                  <div className="mt-2 text-[11px] text-white/55">Slider only.</div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Advanced sampling */}
      <AnimatePresence initial={false}>
        {showAdvanced && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className={cx("mt-1", ui.surface2, ui.insetStroke, "p-4")}>
              <div className="mb-3 text-xs font-semibold text-white/85">
                Advanced sampling
              </div>

              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between text-xs text-white/70">
                    <span>Top-k</span>
                    <span className="font-medium text-white/90">{topK}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={200}
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                    className="mt-2 w-full accent-white"
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between text-xs text-white/70">
                    <span>Temperature</span>
                    <span className="font-medium text-white/90">
                      {temperature.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.6}
                    max={1.4}
                    step={0.01}
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="mt-2 w-full accent-white"
                  />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Generate / Reset */}
      <div className="pt-1 space-y-2">
        <button
          onClick={startGenerate}
          disabled={isGenerating}
          className={cx(
            "group relative w-full overflow-hidden rounded-3xl px-4 py-3 text-sm font-semibold ring-1 transition",
            "shadow-[0_18px_28px_-20px_rgba(0,0,0,0.9)]",
            isGenerating
              ? "bg-white/[0.10] text-white/60 ring-white/10"
              : "bg-white text-black ring-white/15 hover:brightness-105"
          )}
          type="button"
        >
          <div className="relative z-10 flex items-center justify-center gap-2">
            {isGenerating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            {isGenerating ? "Generating…" : "Generate"}
          </div>
        </button>

        <button
          onClick={resetQueue}
          className={cx(ui.btnBase, ui.btnGhost, "w-full")}
          type="button"
        >
          <Trash2 className="h-4 w-4 text-white/80" /> Reset
        </button>

        <div className="pt-1 text-[11px] text-white/55">
          Output defaults to{" "}
          <span className="text-white/80 font-medium">2 samples</span>.
        </div>
      </div>
    </div>
  );

  const ControlsSidebar = (
    <aside
      className={cx(
        "col-span-1 min-w-0 md:col-span-4 lg:col-span-3 self-start",
        ui.surface1,
        ui.insetStroke,
        "p-4",
        // Keep the controls column independently scrollable on desktop.
        "md:max-h-[calc(100vh-96px)] md:overflow-auto"
      )}
    >
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold tracking-tight">Controls</div>
          <div className="mt-1 text-[11px] text-white/55">
            Settings • Sampling • Generate
          </div>
        </div>

        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className={cx(ui.btnBase, ui.btnGhost)}
          type="button"
        >
          <SlidersHorizontal className="h-4 w-4 text-white/80" />
          {showAdvanced ? "Hide" : "Advanced"}
        </button>
      </div>

      <div className="mt-4">{ControlsCore}</div>
    </aside>
  );

  const ControlsDock = (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.26, ease: easeUI }}
      className={cx(ui.surface1, ui.insetStroke, "p-4")}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Pill active>
            <PanelLeft className="h-4 w-4" />
            Studio Mode
          </Pill>
          <Pill>
            <Clock3 className="h-4 w-4 text-white/80" />
            ETA ~ {formatMmSs(etaSeconds)}
          </Pill>
          <Pill>
            <span className={cx("h-2 w-2 rounded-full", gpuInfo && gpuInfo.idle > 0 ? "bg-emerald-400/80" : "bg-amber-400/80")} />
            {gpuInfo ? `GPU ${gpuInfo.idle}/${gpuInfo.total}` : "GPU"}
          </Pill>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setDockExpanded((v) => !v)}
            className={cx(ui.btnBase, ui.btnGhost)}
            type="button"
          >
            {dockExpanded ? (
              <PanelRight className="h-4 w-4 text-white/80" />
            ) : (
              <PanelLeft className="h-4 w-4 text-white/80" />
            )}
            {dockExpanded ? "Hide controls" : "Show controls"}
          </button>

          <button
            onClick={startGenerate}
            disabled={isGenerating}
            className={cx(
              ui.btnBase,
              isGenerating ? "bg-white/[0.10] text-white/55 ring-white/10" : ui.btnPrimary
            )}
            type="button"
          >
            {isGenerating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            {isGenerating ? "Generating…" : "Generate"}
          </button>

          <button
            onClick={resetQueue}
            className={cx(ui.btnBase, ui.btnGhost)}
            type="button"
          >
            <Trash2 className="h-4 w-4 text-white/80" />
            Reset
          </button>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {dockExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22 }}
            className="overflow-hidden"
          >
            <div className={cx("mt-4", ui.surface2, ui.insetStroke, "p-4")}>
              <div className="flex items-center justify-between">
                <div className="text-[11px] tracking-[0.18em] uppercase text-white/55">
                  Controls
                </div>
                <div className="text-[11px] text-white/55">Expanded</div>
              </div>
              <div className="mt-3">{ControlsCore}</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );

  /** ---------- Prompt panel ---------- */
  const renderPromptPanel = (variant: "compose" | "studio") => (
    <motion.section
      layout
      layoutId="panel-prompt"
      transition={panelTransition}
      className={cx(ui.surface1, ui.insetStroke, "p-5", variant === "studio" && "h-full")}
    >
      <div className="flex items-start justify-between gap-3 pb-4">
        <div>
          <div className="text-sm font-semibold tracking-tight">Prompt</div>
          <div className="mt-1 text-xs text-white/55">
            {promptMode === "natural" ? "Natural language" : "Tag builder"}
          </div>
        </div>
        <Pill active>{promptMode === "natural" ? "Natural" : "Tags"}</Pill>
      </div>

      <Divider />

      <div className="mt-4">
        <AnimatePresence mode="wait" initial={false}>
          {promptMode === "natural" ? (
            <motion.div
              key="prompt-natural"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              transition={{ duration: 0.18 }}
              className={cx(variant === "studio" && "h-full flex flex-col")}
            >
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={variant === "compose" ? 16 : 14}
                placeholder="Describe style, vibe, instruments, tempo, mix…"
                className={cx(
                  "w-full resize-none rounded-2xl px-4 py-3 text-sm text-white/90 outline-none placeholder:text-white/35",
                  variant === "studio" && "flex-1",
                  ui.surface3,
                  ui.insetStroke,
                  "focus:ring-1 focus:ring-white/15 focus:border-white/20"
                )}
                style={variant === "studio"
                  ? { maxHeight: "760px", overflowY: "auto" }
                  : undefined}
              />
            </motion.div>
          ) : (
            <motion.div
              key="prompt-tags"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              transition={{ duration: 0.18 }}
              className="space-y-4"
            >
              <div className={cx(ui.surface2, ui.insetStroke, "p-4")}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold tracking-tight">Tag builder</div>
                    <div className="mt-1 text-xs text-white/55">
                      Pick tags; we join them directly.
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => navigator.clipboard?.writeText(builtPrompt)}
                      className={cx(ui.btnBase, ui.btnSoft)}
                      type="button"
                    >
                      <Copy className="h-4 w-4 text-white/85" /> Copy
                    </button>
                    <button
                      onClick={clearAllTags}
                      className={cx(ui.btnBase, ui.btnGhost)}
                      type="button"
                    >
                      <Trash2 className="h-4 w-4 text-white/80" /> Clear
                    </button>
                  </div>
                </div>

                <div className={cx("mt-4", ui.surface3, ui.insetStroke, "p-4")}>
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-[11px] tracking-[0.18em] uppercase text-white/55">
                      Assembled prompt
                    </div>
                    <div className="text-[11px] text-white/55">
                      Total tags: <span className="text-white/80">{totalSelectedTags}</span>
                    </div>
                  </div>
                  <div className="mt-2 text-sm text-white/90">
                    {builtPrompt.trim() ? (
                      builtPrompt
                    ) : (
                      <span className="text-white/40">No tags selected yet.</span>
                    )}
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-12">
                  <div className="col-span-1 md:col-span-12 lg:col-span-4">
                    <div className={cx(ui.surface3, ui.insetStroke, "p-4")}>
                      <div className="text-[11px] tracking-[0.18em] uppercase text-white/55">
                        Categories
                      </div>
                      <div className="mt-3 grid grid-cols-2 gap-2 lg:grid-cols-1">
                        {CATEGORIES.map((c) => {
                          const tone = CATEGORY_TONE[c];
                          const cls = toneClasses(tone);
                          const active = activeCat === c;
                          return (
                            <button
                              key={c}
                              onClick={() => setActiveCat(c)}
                              className={cx(
                                "rounded-2xl px-3 py-2 text-xs transition text-left",
                                active ? cls.tabOn : cls.tabOff
                              )}
                              type="button"
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-semibold">{CATEGORY_META[c].short}</span>
                                <span className="text-white/55">{selectedTags[c].length}</span>
                              </div>
                              <div className="mt-1 text-[11px] text-white/55 line-clamp-1">
                                {CATEGORY_META[c].hint}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </div>

                  <div className="col-span-1 md:col-span-12 lg:col-span-8 space-y-4">
                    <div className={cx(ui.surface3, ui.insetStroke, "p-4")}>
                      <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-black/15 px-3 py-2">
                        <Search className="h-4 w-4 text-white/55" />
                        <input
                          value={tagQuery}
                          onChange={(e) => setTagQuery(e.target.value)}
                          placeholder={`Search in ${activeCat}…`}
                          className="w-full bg-transparent text-xs text-white/85 outline-none placeholder:text-white/35"
                        />
                      </div>

                      <div className="mt-4 max-h-56 overflow-auto pr-1">
                        <div className="flex flex-wrap gap-2">
                          {visibleTags.map((t) => {
                            const selected = selectedTags[t.category].includes(t.label);
                            return (
                              <TagChip
                                key={t.category + ":" + t.label}
                                label={t.label}
                                selected={selected}
                                tone={CATEGORY_TONE[t.category]}
                                onClick={() => toggleTag(t.category, t.label)}
                              />
                            );
                          })}
                        </div>
                        {visibleTags.length === 0 && (
                          <div className="mt-3 text-xs text-white/55">No matches.</div>
                        )}
                      </div>

                      <div className="mt-4 flex items-center justify-between">
                        <div className="text-[11px] text-white/55">
                          Keep 3–8 tags for clean prompts.
                        </div>
                        <div className="flex items-center gap-2">
                          {!tagQuery.trim() && (
                            <button
                              onClick={shuffleTags}
                              className={cx(ui.btnBase, ui.btnGhost, "text-[11px]")}
                              type="button"
                            >
                              <RefreshCw className="h-3 w-3" />
                              {canShuffle ? "More tags" : "Back to top"}
                            </button>
                          )}
                          <button
                            onClick={() => clearCategory(activeCat)}
                            className={cx(ui.btnBase, ui.btnGhost, "text-[11px]")}
                            type="button"
                          >
                            Clear {CATEGORY_META[activeCat].short}
                          </button>
                        </div>
                      </div>
                    </div>

                    <div className={cx(ui.surface3, ui.insetStroke, "p-4")}>
                      <div className="text-[11px] tracking-[0.18em] uppercase text-white/55">
                        Selected
                      </div>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {CATEGORIES.flatMap((c) =>
                          selectedTags[c].map((label) => (
                            <SelectedTagPill
                              key={`${c}:${label}`}
                              label={label}
                              tone={CATEGORY_TONE[c]}
                              onRemove={() => removeTag(c, label)}
                            />
                          ))
                        )}
                        {totalSelectedTags === 0 && (
                          <span className="text-xs text-white/45">None</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.section>
  );

  /** ---------- Lyrics panel (Compose only; hidden in instrumental but preserved) ---------- */
  const LyricsPanelCompose = (
    <motion.section
      layout
      layoutId="panel-lyrics"
      transition={panelTransition}
      initial={{ opacity: 0, x: 22, scale: 0.99 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 22, scale: 0.99 }}
      className={cx(ui.surface1, ui.insetStroke, "p-5")}
    >
      <div className="flex items-start justify-between gap-3 pb-4">
        <div>
          <div className="text-sm font-semibold tracking-tight">Lyrics</div>
          <div className="mt-1 text-xs text-white/55">Auto-grow to max height, then scroll</div>
        </div>
        <Pill active>Required</Pill>
      </div>

      <Divider />

      <textarea
        ref={lyricsRef}
        value={lyrics}
        onChange={(e) => setLyrics(e.target.value)}
        rows={20}
        className={cx(
          "mt-4 w-full resize-none rounded-2xl px-4 py-3 text-sm text-white/90 outline-none placeholder:text-white/35",
          ui.surface3,
          ui.insetStroke,
          "focus:ring-1 focus:ring-white/15 focus:border-white/20"
        )}
        style={{ maxHeight: `${LYRICS_MAX_PX_COMPOSE}px` }}
        placeholder="Paste lyrics here…"
      />
      <div className="mt-2 text-[11px] text-white/55">
        Switching to Instrumental hides this panel but keeps the text.
      </div>
    </motion.section>
  );

  /** ---------- Studio: collapsible input ---------- */
  const PromptAndLyricsStudioAccordion = (
    <div className={cx(ui.surface1, ui.insetStroke, "p-5")}>
      <button
        onClick={() => setInputOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3"
        type="button"
      >
        <div className="flex items-center gap-2">
          <Pill active>
            <PanelLeft className="h-4 w-4" />
            Input
          </Pill>
          <div className="text-left">
            <div className="text-sm font-semibold tracking-tight">Prompt & Lyrics</div>
            <div className="mt-0.5 text-xs text-white/55">
              Edit inputs and regenerate.
            </div>
          </div>
        </div>
        <div className="inline-flex items-center gap-2 text-xs text-white/70">
          {inputOpen ? (
            <>
              <ChevronUp className="h-4 w-4" /> Collapse
            </>
          ) : (
            <>
              <ChevronDown className="h-4 w-4" /> Expand
            </>
          )}
        </div>
      </button>

      <AnimatePresence initial={false}>
        {inputOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22 }}
            className="overflow-hidden"
          >
            <div className="mt-4">
              <Divider />
            </div>

            <div
              className={cx(
                "mt-5 grid grid-cols-1 gap-5",
                mode === "vocal" && "lg:grid-cols-12"
              )}
            >
              <div className={cx(mode === "vocal" && "lg:col-span-7 h-full")}>
                {renderPromptPanel("studio")}
              </div>

              <AnimatePresence mode="wait" initial={false}>
                {mode === "vocal" ? (
                  <motion.section
                    key="studio-lyrics"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    transition={{ duration: 0.2 }}
                    className={cx(ui.surface1, ui.insetStroke, "h-full p-5 flex flex-col lg:col-span-5")}
                  >
                    <div>
                      <div className="text-sm font-semibold tracking-tight">Lyrics</div>
                      <div className="mt-1 text-xs text-white/55">
                        Scrollable view for long lyrics.
                      </div>
                    </div>

                    <div className="mt-4">
                      <Divider />
                    </div>

                    <textarea
                      value={lyrics}
                      onChange={(e) => setLyrics(e.target.value)}
                      rows={14}
                      placeholder="Paste lyrics here…"
                      className={cx(
                        "mt-4 w-full flex-1 resize-none rounded-2xl px-4 py-3 text-sm text-white/90 outline-none placeholder:text-white/35",
                        ui.surface3,
                        ui.insetStroke,
                        "focus:ring-1 focus:ring-white/15 focus:border-white/20"
                      )}
                      style={{ maxHeight: "760px", overflowY: "auto" }}
                    />
                  </motion.section>
                ) : (
                  <motion.div
                    key="studio-nolyrics"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    transition={{ duration: 0.2 }}
                  >
                    <div className={cx(ui.surface2, ui.insetStroke, "p-4")}>
                      <div className="text-sm font-semibold tracking-tight">
                        Instrumental mode
                      </div>
                      <div className="mt-1 text-xs text-white/55">
                        Lyrics are hidden; content is preserved.
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  /** ---------- Results section ---------- */
  const readyCount = tracks.filter((t) => t.status === "ready").length;
  const busyCount = tracks.filter((t) => t.status !== "ready").length;

  const ResultsSection = (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: easeUI }}
      className={cx(ui.surface1, ui.insetStroke, "p-5")}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold tracking-tight">Generation results</div>
          <div className="mt-1 text-xs text-white/55">
            {readyCount} ready • {busyCount} in progress
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Pill>
            <span className="text-white/70">samples</span>{" "}
            <span className="font-semibold">{tracks.length}</span>
          </Pill>
          <Pill>
            <span className="text-white/70">target</span>{" "}
            <span className="font-semibold">{durationMin}m</span>
          </Pill>
        </div>
      </div>

      <div className="mt-4">
        <Divider />
      </div>

      <div className="mt-5 space-y-4">
        {tracks.map((t) => {
          const loadingPhase = isLoadingPhase(t.phaseDisplay, t.workerPhase);
          const visibleProgress = getVisibleProgress(t.progress, displayProgressByTrack[t.id]);
          return (
            <motion.div
              key={t.id}
              layout
              className={cx(
                ui.surface2,
                ui.insetStroke,
                "p-5 transition",
                "hover:bg-white/[0.10]"
              )}
            >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <div className="text-sm font-semibold tracking-tight">{t.title}</div>

                  {t.status === "ready" || t.status === "done" ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-400/10 px-2 py-1 text-[11px] text-emerald-200 ring-1 ring-emerald-300/20">
                      <CheckCircle2 className="h-3.5 w-3.5" /> Ready
                    </span>
                  ) : t.status === "error" ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-red-400/10 px-2 py-1 text-[11px] text-red-200 ring-1 ring-red-300/20">
                      <X className="h-3.5 w-3.5" /> Error
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-white/[0.06] px-2 py-1 text-[11px] text-white/70 ring-1 ring-white/10">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      {t.phaseDisplay || (t.status === "queued" ? "Queued"
                        : t.status === "superres" ? "Super-res"
                        : "Generating")}
                    </span>
                  )}
                </div>
                <div className="mt-1 text-xs text-white/55">
                  {(t.status === "ready" || t.status === "done") && t.actualSeconds
                    ? `Duration • ${formatMmSs(t.actualSeconds)}`
                    : `Target • ${formatMmSs(t.seconds)}`}
                </div>
                {(t.status === "ready" || t.status === "done") && (
                  <div className="mt-1 text-xs text-white/45">
                    {formatMmSs(playbackByTrack[t.id]?.currentTime ?? 0)} / {formatMmSs(
                      playbackByTrack[t.id]?.duration || t.actualSeconds || t.seconds
                    )}
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    if (t.status !== "ready" && t.status !== "done") return;
                    const isActive = activePreview === t.id;
                    // Pause any currently playing
                    if (activePreview) {
                      const prev = audioRefs.current.get(activePreview);
                      if (prev) prev.pause();
                    }
                    if (isActive) {
                      setActivePreview(null);
                    } else {
                      setActivePreview(t.id);
                      const el = audioRefs.current.get(t.id);
                      if (el) el.play();
                    }
                  }}
                  disabled={t.status !== "ready" && t.status !== "done"}
                  className={cx(
                    ui.btnBase,
                    (t.status !== "ready" && t.status !== "done") ? "bg-white/[0.06] text-white/40 ring-white/10" : ui.btnSoft
                  )}
                  type="button"
                >
                  {activePreview === t.id ? (
                    <Pause className="h-4 w-4" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                  Preview
                </button>

                {t.jobId != null && t.trackIndex != null ? (
                  <a
                    href={(t.status === "ready" || t.status === "done") ? trackWavUrl(t.jobId, t.trackIndex) : "#"}
                    onClick={(e) => { if (t.status !== "ready" && t.status !== "done") e.preventDefault(); }}
                    className={cx(
                      ui.btnBase,
                      (t.status !== "ready" && t.status !== "done") ? "bg-white/[0.06] text-white/40 ring-white/10 pointer-events-none" : ui.btnPrimary
                    )}
                  >
                    <Download className="h-4 w-4" /> Export
                  </a>
                ) : (
                  <button
                    disabled
                    className={cx(ui.btnBase, "bg-white/[0.06] text-white/40 ring-white/10")}
                    type="button"
                  >
                    <Download className="h-4 w-4" /> Export
                  </button>
                )}
              </div>
            </div>

            <div className="mt-4">
              <div className="h-2 w-full rounded-full bg-white/10">
                {loadingPhase ? (
                  <div className="relative h-2 overflow-hidden rounded-full">
                    <motion.div
                      className="absolute inset-y-0 left-0 w-1/3 rounded-full bg-white/65"
                      animate={{ x: ["-120%", "320%"] }}
                      transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                    />
                  </div>
                ) : (
                  <div
                    className="h-2 rounded-full bg-white/70 transition-opacity duration-500 ease-out"
                    style={{
                      width: `${visibleProgress.toFixed(1)}%`,
                      opacity: (t.status === "ready" || t.status === "done") ? 0.85 : 0.55,
                    }}
                  />
                )}
              </div>
              {/* progress text removed — shown in top-left pill only */}
            </div>

            <div className="mt-4">
              <Waveform
                audioUrl={(t.status === "ready" || t.status === "done") && t.jobId != null && t.trackIndex != null
                  ? trackMp3Url(t.jobId, t.trackIndex) : null}
                isReady={t.status === "ready" || t.status === "done"}
                audioRef={{ current: audioRefs.current.get(t.id) ?? null }}
                isPlaying={activePreview === t.id}
              />
              {(t.status === "ready" || t.status === "done") && t.jobId != null && t.trackIndex != null && (
                <audio
                  ref={(el) => { if (el) audioRefs.current.set(t.id, el); else audioRefs.current.delete(t.id); }}
                  src={trackMp3Url(t.jobId, t.trackIndex)}
                  onLoadedMetadata={(e) => syncTrackPlayback(t.id, e.currentTarget)}
                  onTimeUpdate={(e) => syncTrackPlayback(t.id, e.currentTarget)}
                  onSeeked={(e) => syncTrackPlayback(t.id, e.currentTarget)}
                  onEnded={(e) => {
                    syncTrackPlayback(t.id, e.currentTarget);
                    setActivePreview(null);
                  }}
                  hidden
                />
              )}
            </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );

  /** ---------- Compose layout: Three-panel for vocal, Two-panel for instrumental ---------- */
  const ComposePanels = (
    <motion.div
      layout
      transition={panelTransition}
      className="col-span-1 min-w-0 md:col-span-8 lg:col-span-9"
    >
      <motion.div
        layout
        transition={panelTransition}
        className="grid grid-cols-1 gap-5 items-start min-w-0 md:grid-cols-12"
      >
        {/* Prompt: center if vocal; right if instrumental (takes more width) */}
        <motion.div
          layout
          transition={panelTransition}
          className={cx(
            "col-span-1 min-w-0 md:col-span-12",
            mode === "vocal" ? "lg:col-span-6 xl:col-span-7" : "lg:col-span-12"
          )}
        >
          {renderPromptPanel("compose")}
        </motion.div>

        {/* Lyrics: only in vocal (animated in/out), content preserved regardless */}
        <AnimatePresence initial={false}>
          {mode === "vocal" && (
            <motion.div
              layout
              transition={panelTransition}
              className="col-span-1 min-w-0 md:col-span-12 lg:col-span-5"
            >
              {LyricsPanelCompose}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );

  /** ---------- render ---------- */
  return (
    <div className={cx(ui.appBg, "w-full max-w-full overflow-x-hidden")}>
      {/* Background layers (kept, but tuned for depth) */}
      <div className="pointer-events-none fixed inset-0">
        <div className="absolute inset-0 bg-gradient-to-b from-white/8 via-transparent to-black/20" />
        <div className="absolute inset-0 bg-gradient-to-br from-white/6 via-transparent to-white/4" />
        {/* subtle grain (no external asset) */}
        <div className="absolute inset-0 opacity-[0.05] mix-blend-overlay bg-[radial-gradient(circle_at_1px_1px,rgba(255,255,255,0.16)_1px,transparent_0)] [background-size:22px_22px]" />
      </div>

      {/* Top bar */}
      <header className="relative z-10 border-b border-white/10 bg-white/[0.05] backdrop-blur-xl shadow-[0_10px_30px_-16px_rgba(0,0,0,0.8)]">
        <div className="flex w-full flex-wrap items-center justify-between gap-3 px-4 py-3 md:flex-nowrap md:gap-0 md:px-6 md:py-4 xl:px-10">
          <div className="flex items-center gap-3">
            <div
              className={cx(
                "grid h-8 w-8 shrink-0 place-items-center rounded-2xl",
                "bg-white/[0.10] ring-1 ring-white/14",
                "shadow-[0_10px_22px_-18px_rgba(0,0,0,0.9)]"
              )}
            >
              <AudioLines className="h-5 w-5" />
            </div>
            <div className="leading-tight">
              <div className="flex flex-wrap items-center gap-2 md:gap-3">
                <div className="text-lg font-semibold tracking-[-0.025em] md:text-[22px]">
                  <span className="bg-gradient-to-b from-white to-white/70 bg-clip-text text-transparent">
                    Khala
                  </span>
                </div>
                <div className="hidden h-5 w-px bg-white/12 md:block" />
                <div className="flex flex-wrap items-center gap-2">
                  <span className="hidden text-[20px] font-medium text-white/80 md:inline">AI Music Studio</span>
                  <span className="hidden text-white/35 md:inline">•</span>
                  <span className="inline-flex items-center rounded-full px-2 py-0.5 text-xs md:text-[15px]
                    bg-white/[0.10] text-white/80 ring-1 ring-white/12">
                    v1.0
                  </span>
                  <span className="inline-flex items-center rounded-full px-2 py-0.5 text-xs md:text-[15px]
                    bg-emerald-400/10 text-emerald-200 ring-1 ring-emerald-300/20">
                    Beta
                  </span>
                </div>
              </div>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Pill active>
              {mode === "vocal" ? (
                <Mic2 className="h-4 w-4 md:h-5 md:w-5 text-white/85" />
              ) : (
                <Music2 className="h-4 w-4 md:h-5 md:w-5 text-white/85" />
              )}
              {mode === "vocal" ? "Vocal" : "Instrumental"}
            </Pill>
            <Pill>
              <Clock3 className="h-4 w-4 md:h-5 md:w-5 text-white/80" />
              {formatMmSs(etaSeconds)}
            </Pill>
            <Pill>
              <span className={cx("h-4 w-4 md:h-5 md:w-5 rounded-full", gpuInfo && gpuInfo.idle > 0 ? "bg-emerald-400/80" : "bg-amber-400/80")} />
              {gpuInfo ? `GPU ${gpuInfo.idle}/${gpuInfo.total}` : "GPU"}
            </Pill>
          </div>
        </div>
      </header>

      <main className="relative z-10 w-full overflow-x-hidden px-3 py-4 md:px-6 md:py-6 xl:px-10">
        <div className="mx-auto grid w-full min-w-0 max-w-[1600px] grid-cols-1 gap-5 md:grid-cols-12 md:gap-6 2xl:max-w-[1800px]">
          <AnimatePresence mode="wait" initial={false}>
            {studioMode ? (
              <motion.div
                key="studio"
                initial={{ opacity: 0, y: 12, filter: "blur(2px)" as any }}
                animate={{ opacity: 1, y: 0, filter: "blur(0px)" as any }}
                exit={{ opacity: 0, y: -10, filter: "blur(2px)" as any }}
                transition={{ duration: 0.32, ease: easeUI }}
                className="col-span-1 md:col-span-12 space-y-5"
              >
                <div className="sticky top-3 z-20">
                  {ControlsDock}
                  <div className="h-4" />
                </div>

                {PromptAndLyricsStudioAccordion}
                {ResultsSection}

                <div className="pb-10" />
              </motion.div>
            ) : (
              <motion.div
                key="compose"
                initial={{ opacity: 0, y: 12, filter: "blur(2px)" as any }}
                animate={{ opacity: 1, y: 0, filter: "blur(0px)" as any }}
                exit={{ opacity: 0, y: -10, filter: "blur(2px)" as any }}
                transition={{ duration: 0.32, ease: easeUI }}
                className="col-span-1 grid grid-cols-1 gap-5 items-start min-w-0 md:col-span-12 md:grid-cols-12"
              >
                {ControlsSidebar}
                {ComposePanels}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
