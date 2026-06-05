import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const tamilDictionary = {
  'person': 'முன்னால் ஒருவர் இருக்கிறார்',
  'car': 'முன்னால் ஒரு கார் உள்ளது',
  'motorcycle': 'முன்னால் ஒரு பைக் உள்ளது',
  'bicycle': 'முன்னால் ஒரு சைக்கிள் உள்ளது',
  'bus': 'முன்னால் ஒரு பஸ் உள்ளது',
  'truck': 'முன்னால் ஒரு லாரி உள்ளது',
  'traffic light': 'முன்னால் ட்ராஃபிக் சிக்னல் உள்ளது',
  'stop sign': 'முன்னால் ஒரு அறிவிப்பு பலகை உள்ளது',
  'dog': 'முன்னால் ஒரு நாய் உள்ளது',
  'cat': 'முன்னால் ஒரு பூனை உள்ளது',
  'cow': 'முன்னால் ஒரு மாடு உள்ளது',
  'chair': 'முன்னால் ஒரு நாற்காலி உள்ளது',
  'bench': 'முன்னால் ஒரு பெஞ்ச் உள்ளது',
  'potted plant': 'முன்னால் ஒரு செடி உள்ளது',
  'bottle': 'முன்னால் ஒரு பாட்டில் உள்ளது',
  'cell phone': 'முன்னால் ஒரு செல்போன் உள்ளது',
  'laptop': 'முன்னால் ஒரு லேப்டாப் உள்ளது',
};

// ORS maneuver "type" code -> Tamil phrase.
const TAMIL_MANEUVERS = {
  0: 'இடதுபுறம் திரும்பவும் (Turn Left)',                
  1: 'வலதுபுறம் திரும்பவும் (Turn Right)',              
  2: 'கூர்மையாக இடதுபுறம் திரும்பவும் (Sharp Left)',        
  3: 'கூர்மையாக வலதுபுறம் திரும்பவும் (Sharp Right)',        
  4: 'சற்று இடதுபுறம் செல்லவும் (Slight Left)',              
  5: 'சற்று வலதுபுறம் செல்லவும் (Slight Right)',              
  6: 'நேராக செல்லவும் (Continue Straight)',               
  7: 'சுற்றுவட்டத்தில் நுழையவும் (Enter Roundabout)',             
  8: 'சுற்றுவட்டத்திலிருந்து வெளியேறவும் (Exit Roundabout)',      
  9: 'பின்னோக்கி திரும்பவும் (U-Turn)',                 
  10: 'நீங்கள் இலக்கை அடைந்துவிட்டீர்கள் (Arrive)',      
  11: 'நேராக நடக்கத் தொடங்கவும் (Depart)',              
  12: 'இடதுபக்கம் ஒட்டி செல்லவும் (Keep Left)',            
  13: 'வலதுபக்கம் ஒட்டி செல்லவும் (Keep Right)',            
};

const ANNOUNCE_RADIUS_M = 20;

function distanceMeters(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const lastSpokenTimeRef = useRef({});
  const lastDistRef = useRef(Infinity); 
  const warned20mRef = useRef(false);   
  const warned5mRef = useRef(false);    

  const [isHighContrast, setIsHighContrast] = useState(false);
  const [model, setModel] = useState(null);
  const [isVoiceOn, setIsVoiceOn] = useState(localStorage.getItem('voiceEnabled') === 'true');
  const [aiSight, setAiSight] = useState('Waiting to start...');
  const [tamilVoice, setTamilVoice] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [navStatus, setNavStatus] = useState('');
  const lastSpokenDistanceRef = useRef(null);

  const [destination, setDestination] = useState('');
  const [navigating, setNavigating] = useState(false);
  const stepsRef = useRef([]);        
  const stepIndexRef = useRef(0);     
  const watchIdRef = useRef(null);    

  const isVoiceOnRef = useRef(isVoiceOn);
  const slowLoopSpeakingRef = useRef(false);

  useEffect(() => {
    cocoSsd.load().then((loadedModel) => {
      setModel(loadedModel);
      setAiSight('Fast Brain AI Model loaded!');
    });

    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      const foundVoice = voices.find((v) => v.lang === 'ta-IN' || v.lang.startsWith('ta'));
      if (foundVoice) setTamilVoice(foundVoice);
    };
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;

    return () => {
      if (watchIdRef.current != null) navigator.geolocation.clearWatch(watchIdRef.current);
    };
  }, []);

  const toggleVoice = () => {
    const newSetting = !isVoiceOn;
    setIsVoiceOn(newSetting);
    isVoiceOnRef.current = newSetting;
    localStorage.setItem('voiceEnabled', newSetting.toString());
  };

  const isMobile = typeof navigator !== 'undefined' && /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
  const mobileRateFactor = 0.6;
  const normalizeRate = (rate) => (isMobile ? rate * mobileRateFactor : rate);

  const speakTamil = (textToSpeak, { priority = false, rate = 1.5 } = {}) => {
    if (!textToSpeak) return;
    if (priority) {
      window.speechSynthesis.cancel();
      slowLoopSpeakingRef.current = false;
    } else if (window.speechSynthesis.speaking) {
      return;
    }
    const msg = new SpeechSynthesisUtterance(textToSpeak);
    msg.rate = normalizeRate(rate);
    msg.pitch = 1.1;
    if (tamilVoice) msg.voice = tamilVoice;
    else msg.lang = 'ta-IN';
    window.speechSynthesis.speak(msg);
  };

  const startCamera = async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    }
  };

  const detectObjects = async () => {
    if (model && videoRef.current && videoRef.current.readyState === 4) {
      const predictions = await model.detect(videoRef.current);
      
      if (predictions.length > 0) {
        const firstObj = predictions[0];
        const boxWidth = Math.round(firstObj.bbox[2]);
        setAiSight(`Fast Brain sees: ${firstObj.class} (Width: ${boxWidth}px)`);
        
        if (tamilDictionary[firstObj.class] && boxWidth > 150 && isVoiceOnRef.current) {
          const now = Date.now();
          const cooldownTime = 5000;
          const lastTimeSpoken = lastSpokenTimeRef.current[firstObj.class] || 0;

          if (now - lastTimeSpoken > cooldownTime) {
            const isPriority = slowLoopSpeakingRef.current;
            speakTamil(tamilDictionary[firstObj.class], { priority: isPriority, rate: 1.8 });
            lastSpokenTimeRef.current[firstObj.class] = now;
          }
        }
      } else {
        setAiSight('Path clear.');
      }
    }
    requestAnimationFrame(detectObjects);
  };

  const triggerDeepScan = async () => {
    if (!videoRef.current || videoRef.current.readyState !== 4) return;
    setIsScanning(true);
    setAiSight('Slow Brain (Groq) is thinking...');
    speakTamil('ஆராய்கிறேன்', { priority: true, rate: 1.8 });

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const maxWidth = 640;
    const scale = maxWidth / video.videoWidth;
    canvas.width = maxWidth;
    canvas.height = video.videoHeight * scale;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64Image = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];

    try {
      const response = await fetch('https://divya-drishti-the-echowalk-phase-1-demo.onrender.com/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: base64Image }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      const resultText = data.tamil_text || 'மன்னிக்கவும்';
      setAiSight(`Groq says: ${resultText}`);

      window.speechSynthesis.cancel();
      const resultMsg = new SpeechSynthesisUtterance(resultText);
      resultMsg.rate = normalizeRate(1.5);
      if (tamilVoice) resultMsg.voice = tamilVoice;
      else resultMsg.lang = 'ta-IN';
      slowLoopSpeakingRef.current = true;
      resultMsg.onend = () => { slowLoopSpeakingRef.current = false; };
      window.speechSynthesis.speak(resultMsg);
    } catch (error) {
      console.error('Backend Error Details:', error);
      speakTamil('தொடர்பு கொள்ள முடியவில்லை.', { priority: true });
    }
    setIsScanning(false);
  };

  const handleLocationAwareness = () => {
    speakTamil("உங்கள் இடத்தை தேடுகிறேன்...", { priority: true, rate: 1.8 }); 
    setAiSight("Finding GPS Location...");

    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(async (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;

        try {
          const response = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}`);
          const data = await response.json();
          const addr = data.address;
          const locationName = addr.suburb || addr.neighbourhood || addr.residential || addr.village || addr.road || addr.town || "கோயம்புத்தூர்";         
          const locationMessage = `நீங்கள் இப்போது ${locationName} பகுதியில் உள்ளீர்கள்.`; 
          
          setAiSight(`Location: ${locationName}`);
          speakTamil(locationMessage, { priority: true });

        } catch (error) {
          console.error("GPS API Error:", error);
          speakTamil("இடத்தை கண்டறிய முடியவில்லை.", { priority: true }); 
        }
      }, (error) => {
        console.error("GPS Permission Error:", error);
        speakTamil("ஜிபிஎஸ் அனுமதி தேவை.", { priority: true }); 
      }, {
        enableHighAccuracy: true, timeout: 10000, maximumAge: 0
      });
    } else {
      speakTamil("உங்கள் சாதனத்தில் ஜிபிஎஸ் இல்லை.", { priority: true }); 
    }
  };

  const stopNavigation = (announce = false) => {
    if (watchIdRef.current != null) {
      navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
    }
    setNavigating(false);
    setNavStatus(''); 
    if (announce) speakTamil('வழிகாட்டுதல் நிறுத்தப்பட்டது', { priority: true }); 
  };

  const onPositionUpdate = (pos) => {
    const steps = stepsRef.current;
    const idx = stepIndexRef.current;
    if (idx >= steps.length) return;

    const next = steps[idx];
    const d = Math.round(distanceMeters(pos.coords.latitude, pos.coords.longitude, next.lat, next.lon));
    
    const maneuverTamil = TAMIL_MANEUVERS[next.type] || 'நேராக செல்லவும்';
    
    // VISUAL UPDATE: Shows distance + the actual turn text
    setNavStatus(`${d} மீட்டர்: ${maneuverTamil}`);

    // LOGIC: Wrong Way Detection
    if (lastDistRef.current !== Infinity && d > lastDistRef.current + 10) {
      speakTamil("நீங்கள் தவறான திசையில் செல்கிறீர்கள்", { priority: true, rate: 1.5 }); 
      lastDistRef.current = d; 
    } else if (d < lastDistRef.current) {
      lastDistRef.current = d; 
    }

    // LOGIC: Proximity Triggers
    if (d <= 20 && d > 5 && !warned20mRef.current) {
      speakTamil(`20 மீட்டரில் ${maneuverTamil}`, { priority: true }); 
      warned20mRef.current = true;
    } 
    else if (d <= 5 && !warned5mRef.current) {
      warned5mRef.current = true;
      
      if (next.type === 10 || idx === steps.length - 1) {
        speakTamil("நீங்கள் இலக்கை அடைந்துவிட்டீர்கள். கதவை கண்டுபிடிக்க டீப் ஸ்கேன் பயன்படுத்தவும்.", { priority: true }); 
        stopNavigation(false);
      } else {
        speakTamil(`இப்போது ${maneuverTamil}`, { priority: true }); 
        stepIndexRef.current = idx + 1;
        warned20mRef.current = false;
        warned5mRef.current = false;
        lastDistRef.current = Infinity;
        lastSpokenDistanceRef.current = null;
      }
    } 
    else if (d > 20) {
      if (lastSpokenDistanceRef.current === null || Math.abs(lastSpokenDistanceRef.current - d) >= 15) {
        speakTamil(`இன்னும் ${d} மீட்டர்`, { priority: false, rate: 1.5 });
        lastSpokenDistanceRef.current = d;
      }
    }
  };  

  const startNavigation = () => {
    if (navigating) { stopNavigation(true); return; }

    if (!destination.trim()) {
      speakTamil('இலக்கை உள்ளிடவும்', { priority: true }); 
      return;
    }
    if (!('geolocation' in navigator)) {
      speakTamil('உங்கள் சாதனத்தில் ஜிபிஎஸ் இல்லை', { priority: true });
      return;
    }

    speakTamil('வழியைத் தேடுகிறேன்', { priority: true }); 
    setNavStatus('Finding route...');

    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const startLat = pos.coords.latitude;
        const startLon = pos.coords.longitude;
        try {
          const geo = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(destination)}`
          );
          const geoData = await geo.json();
          if (!geoData.length) {
            speakTamil('இலக்கை கண்டறிய முடியவில்லை', { priority: true });
            setNavStatus('Destination not found');
            return;
          }
          const endLat = parseFloat(geoData[0].lat);
          const endLon = parseFloat(geoData[0].lon);

          const res = await fetch('https://divya-drishti-the-echowalk-phase-1-demo.onrender.com/route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              start_lat: startLat, start_lon: startLon,
              end_lat: endLat, end_lon: endLon,
            }),
          });
          
          if (!res.ok) {
            const e = await res.json().catch(() => ({}));
            throw new Error(e.detail || `HTTP ${res.status}`);
          }
          
          const data = await res.json();
          stepsRef.current = data.steps || [];
          stepIndexRef.current = 0;

          if (!stepsRef.current.length) {
            speakTamil('வழி கிடைக்கவில்லை', { priority: true });
            setNavStatus('Route failed'); // <-- FIX: Update UI if no steps
            return;
          }

          setNavigating(true);
          
          lastSpokenDistanceRef.current = null;
          lastDistRef.current = Infinity;
          warned20mRef.current = false;
          warned5mRef.current = false;
          
          stepIndexRef.current = 1; 

          speakTamil(
            `மொத்த தூரம் ${data.total_distance_m} மீட்டர். ` +
            (TAMIL_MANEUVERS[stepsRef.current[0].type] || 'நேராக நடக்கத் தொடங்கவும்'),
            { priority: true }
          );

          watchIdRef.current = navigator.geolocation.watchPosition(
            onPositionUpdate,
            (err) => console.error('watchPosition error:', err),
            { enableHighAccuracy: true, maximumAge: 1000, timeout: 15000 }
          );
          
        } catch (err) {
          console.error('Route error:', err);
          speakTamil('வழி கண்டறிய முடியவில்லை', { priority: true });
          setAiSight('Route calculation error');
          setNavStatus('Route failed'); // <-- FIX: Clears the "Finding route..." state
        }
      },
      (err) => {
        console.error('GPS error:', err);
        speakTamil('ஜிபிஎஸ் அனுமதி தேவை', { priority: true });
        setNavStatus('GPS Error');
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  };

  return (
    <div style={{ padding: '20px', textAlign: 'center', fontFamily: 'sans-serif' }}>
      <h1>The EchoWalk (Phase 1)</h1>

      <button onClick={toggleVoice} style={{ padding: '10px', margin: '5px' }}>
        Voice Warnings: {isVoiceOn ? 'ON' : 'OFF'}
      </button>

      <button 
        onClick={() => setIsHighContrast(!isHighContrast)} 
        style={{ padding: '10px', margin: '5px', backgroundColor: isHighContrast ? '#333' : '#ddd', color: isHighContrast ? '#fff' : '#000', fontWeight: 'bold' }}
      >
        AR Filter: {isHighContrast ? 'ON' : 'OFF'}
      </button>

      <br /><br />

      <button
        onClick={() => { startCamera(); detectObjects(); }}
        style={{ padding: '10px', backgroundColor: '#90EE90', fontWeight: 'bold', margin: '5px' }}
      >
        1. Start Camera (Fast Brain)
      </button>

      <button
        onClick={triggerDeepScan}
        disabled={isScanning}
        style={{ padding: '15px', margin: '5px', backgroundColor: '#87CEFA', fontWeight: 'bold', fontSize: '18px', borderRadius: '8px' }}
      >
        {isScanning ? 'Thinking...' : '2. DEEP SCAN (Slow Brain)'}
      </button>

      <button 
        onClick={handleLocationAwareness} 
        style={{ padding: '15px', margin: '5px', backgroundColor: '#FFD700', fontWeight: 'bold', fontSize: '18px', borderRadius: '8px' }}
      >
        3. WHERE AM I? (Location)
      </button>

      <br /><br />

      <input
        value={destination}
        onChange={(e) => setDestination(e.target.value)}
        placeholder="Destination (e.g. Coimbatore Junction)"
        style={{ padding: '10px', fontSize: '16px', width: '280px', borderRadius: '8px', border: '1px solid #999' }}
      />
      <button
        onClick={startNavigation}
        style={{ padding: '15px', margin: '10px', backgroundColor: navigating ? '#FF7F7F' : '#FF8C00', fontWeight: 'bold', fontSize: '18px', color: navigating ? 'black' : 'white', borderRadius: '8px' }}
      >
        {navigating ? 'STOP Navigation' : '4. NAVIGATE (Turn-by-Turn)'}
      </button>

      <br />
      
      <h3 style={{ color: 'blue', minHeight: '30px', margin: '5px' }}>{aiSight}</h3>
      <h3 style={{ color: 'green', minHeight: '30px', margin: '5px' }}>{navStatus}</h3>

      <video 
        ref={videoRef} 
        width="400" 
        height="300" 
        style={{ 
          border: '2px solid black',
          borderRadius: '8px',
          transition: 'all 0.3s ease',
          filter: isHighContrast ? 'contrast(300%) grayscale(100%) invert(100%)' : 'none' 
        }} 
        muted 
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}

export default App;