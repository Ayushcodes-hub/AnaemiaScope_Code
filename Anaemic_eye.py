"""
AnaemiaScope - India-Adapted Research Prototype
For IRB-approved clinical studies only. Not for diagnostic use.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import json
from datetime import datetime


# ─── India-Specific Constants ────────────────────────────────────────────────

class IndiaContext:
    """India-specific adaptations for the research prototype."""
    
    # Supported languages
    LANGUAGES = {
        "en": "English",
        "hi": "हिन्दी",
        "ta": "தமிழ்",
        "te": "తెలుగు",
        "bn": "বাংলা"
    }
    
    # Normal Hb ranges for Indian populations (g/dL) - ICMR guidelines
    HB_NORMAL_INDIA = {
        "adult_male": (13.0, 17.0),
        "adult_female_non_pregnant": (12.0, 15.0),
        "adult_female_pregnant": (11.0, 14.0),
        "child_1_5": (11.0, 14.0),
        "child_6_12": (11.5, 15.0),
        "adolescent_male": (12.0, 16.0),
        "adolescent_female": (11.5, 15.0)
    }
    
    # Common confounders in India
    CONFOUNDERS = [
        "conjunctival_jaundice",
        "pterygium",
        "allergic_conjunctivitis",
        "vitamin_a_deficiency",
        "iron_overload",
        "lead_toxicity"
    ]


@dataclass
class IndiaAdaptedResult:
    """Result structure with India-specific fields."""
    pallor_index: float
    estimated_hb: float
    confidence: float
    risk_level: str
    risk_level_hi: str
    calibration_quality: str
    demographic_adjusted_hb: float
    confounders_detected: list
    referral_required: bool
    ayushman_bharat_format: Dict


# ─── Indian Skin Optimised Calibrator (FIXED) ────────────────────────────────

class IndianSkinOptimisedCalibrator:
    """Optimised for Indian skin tones (Fitzpatrick IV-VI)."""
    
    INDIAN_SCLERAL_REFERENCE = {
        "fitzpatrick_iv": {"xy": (0.315, 0.332), "tolerance": 0.05},
        "fitzpatrick_v": {"xy": (0.320, 0.335), "tolerance": 0.06},
        "fitzpatrick_vi": {"xy": (0.325, 0.338), "tolerance": 0.07},
    }
    
    CONJUNCTIVAL_BASELINE_INDIA = {
        "a_star_min": 125.0,
        "a_star_max": 162.0,
        "luminance_min": 55.0,
        "luminance_max": 195.0
    }
    
    def estimate_fitzpatrick_type(self, img_bgr: np.ndarray) -> str:
        """Estimate skin type using ITA method."""
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(img_lab)
        
        median_L = np.median(L)
        median_b = np.median(b)
        
        if median_b < 1:
            ita = 90
        else:
            ita = np.arctan((median_L - 50) / median_b) * 180 / np.pi
        
        if ita > 41:
            return "fitzpatrick_iv"
        elif ita > 28:
            return "fitzpatrick_iv"
        elif ita > 10:
            return "fitzpatrick_v"
        else:
            return "fitzpatrick_vi"
    
    def extract_sclera_india(self, img_bgr: np.ndarray, fitz_type: str) -> Optional[np.ndarray]:
        """Extract sclera with ethnicity-adjusted thresholds."""
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(img_lab)
        
        if fitz_type in ["fitzpatrick_v", "fitzpatrick_vi"]:
            mask = (L > 160) & (np.abs(a.astype(int) - 128) < 18) & (np.abs(b.astype(int) - 128) < 25)
        else:
            mask = (L > 175) & (np.abs(a.astype(int) - 128) < 14) & (np.abs(b.astype(int) - 128) < 18)
        
        mask = mask.astype(np.uint8) * 255
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if mask.sum() < 500:
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pixels = img_rgb[mask > 0]
        return np.median(pixels, axis=0)


# ─── Deep Learning Segmenter (FIXED - removed undefined variable) ────────────

class DeepLearningSegmenter:
    """
    Conjunctival segmentation using classical CV (DL optional).
    Fixed: removed undefined 'use_dl' variable reference.
    """
    
    def __init__(self, use_deep_learning: bool = False, model_path: Optional[str] = None):
        self.use_deep_learning = use_deep_learning
        self.model = None
        
        if use_deep_learning and model_path:
            try:
                # Placeholder for future TensorFlow Lite integration
                # import tflite_runtime.interpreter as tflite
                # self.interpreter = tflite.Interpreter(model_path=model_path)
                print("Deep learning mode selected - model loading placeholder")
            except ImportError:
                print("TensorFlow Lite not available. Falling back to classical CV.")
                self.use_deep_learning = False
    
    def segment(self, corrected_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Segment conjunctiva using DL if available, else classical."""
        if self.use_deep_learning and self.model:
            # DL-based segmentation placeholder
            pass
        
        return self._classical_segmentation(corrected_rgb)
    
    def _classical_segmentation(self, corrected_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Classical segmentation with India-tuned parameters."""
        img_uint8 = (corrected_rgb * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        
        L, a, b = cv2.split(img_lab)
        
        # India-adjusted thresholds
        mask = (
            (L > 55) & (L < 195) &
            (a > 125) & (a < 175) &
            (b > 100) & (b < 165)
        ).astype(np.uint8) * 255
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        if mask.sum() < 200:
            return None
        
        pixels = corrected_rgb[mask > 0]
        return pixels


# ─── Main India-Optimised Pipeline ───────────────────────────────────────────

class AnaemiaScopeIndia:
    """India-optimised anaemia screening research prototype."""
    
    def __init__(self, language: str = "en", use_deep_learning: bool = False):
        self.language = language
        self.calibrator = IndianSkinOptimisedCalibrator()
        self.segmenter = DeepLearningSegmenter(use_deep_learning=use_deep_learning)
        self._analysis_history = []
    
    def analyse(self, img_bgr: np.ndarray, 
                demographic: Optional[Dict] = None) -> IndiaAdaptedResult:
        """Run India-optimised analysis."""
        
        if demographic is None:
            demographic = {"age": 30, "sex": "adult_female_non_pregnant"}
        
        # Step 1: Estimate skin type
        fitz_type = self.calibrator.estimate_fitzpatrick_type(img_bgr)
        sclera = self.calibrator.extract_sclera_india(img_bgr, fitz_type)
        
        # Step 2: Apply calibration
        gains, cct, cal_quality = self._calibrate_with_india_params(sclera, fitz_type)
        
        # Step 3: Colour correction
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        corrected = np.clip(img_rgb * gains[np.newaxis, np.newaxis, :], 0, 1)
        
        # Step 4: Segment conjunctiva
        conjunctiva = self.segmenter.segment(corrected)
        
        if conjunctiva is None or len(conjunctiva) < 100:
            return self._error_result("Conjunctiva not detected", cal_quality)
        
        # Step 5: Compute pallor index
        pallor_index, confidence = self._compute_pallor_india(conjunctiva)
        
        # Step 6: Estimate Hb
        raw_hb = self._pallor_to_hb(pallor_index)
        adjusted_hb = self._demographic_adjustment(raw_hb, demographic)
        
        # Step 7: Risk classification
        risk_level, risk_hi = self._classify_risk_india(adjusted_hb, demographic)
        
        # Step 8: Detect confounders
        confounders = self._detect_confounders(img_bgr, corrected)
        
        # Step 9: Referral recommendation
        referral_needed = self._should_refer(adjusted_hb, risk_level, confounders, confidence)
        
        # Step 10: Format for Ayushman Bharat
        abdm_format = self._format_ayushman_bharat(adjusted_hb, risk_level, confidence)
        
        return IndiaAdaptedResult(
            pallor_index=round(pallor_index, 3),
            estimated_hb=round(raw_hb, 1),
            confidence=round(confidence, 2),
            risk_level=risk_level,
            risk_level_hi=risk_hi,
            calibration_quality=cal_quality,
            demographic_adjusted_hb=round(adjusted_hb, 1),
            confounders_detected=confounders,
            referral_required=referral_needed,
            ayushman_bharat_format=abdm_format
        )
    
    def _calibrate_with_india_params(self, sclera_rgb, fitz_type):
        """Calibration using Indian reference values."""
        if sclera_rgb is None:
            return np.array([1.0, 1.0, 1.0]), 6500.0, "Marginal"
        
        ref = IndianSkinOptimisedCalibrator.INDIAN_SCLERAL_REFERENCE.get(
            fitz_type,
            IndianSkinOptimisedCalibrator.INDIAN_SCLERAL_REFERENCE["fitzpatrick_iv"]
        )
        
        target_white = np.array([ref["xy"][0] / 0.3127, ref["xy"][1] / 0.3290, 0.95])
        target_white = target_white / target_white[1]
        
        gains = np.where(sclera_rgb > 0.02, target_white / sclera_rgb, 1.0)
        gains = gains / gains[1]
        
        quality = "Good" if np.all(gains > 0.7) and np.all(gains < 1.5) else "Marginal"
        return gains, 6500.0, quality
    
    def _compute_pallor_india(self, conjunctiva_pixels):
        """Compute pallor index with India-adjusted bounds."""
        pixels_uint8 = (np.clip(conjunctiva_pixels, 0, 1) * 255).astype(np.uint8)
        pixels_bgr = pixels_uint8[:, ::-1]
        pixels_bgr = pixels_bgr.reshape(-1, 1, 3)
        lab = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3)
        
        a_vals = lab[:, 1].astype(float)
        median_a = np.median(a_vals)
        std_a = np.std(a_vals)
        
        A_MIN = IndianSkinOptimisedCalibrator.CONJUNCTIVAL_BASELINE_INDIA["a_star_min"]
        A_MAX = IndianSkinOptimisedCalibrator.CONJUNCTIVAL_BASELINE_INDIA["a_star_max"]
        
        pallor_index = np.clip((median_a - A_MIN) / (A_MAX - A_MIN), 0.0, 1.0)
        pallor_index = 1.0 - pallor_index
        
        n_pixels = len(a_vals)
        count_confidence = min(1.0, n_pixels / 1500.0)
        consistency = max(0.0, 1.0 - (std_a / 20.0))
        confidence = count_confidence * consistency
        
        return pallor_index, confidence
    
    def _pallor_to_hb(self, pallor_index):
        """Empirical mapping for Indian population."""
        hb = 15.5 - (pallor_index * 8.5)
        return max(4.0, min(17.0, hb))
    
    def _demographic_adjustment(self, raw_hb, demographic):
        """Adjust Hb estimate for demographics."""
        sex = demographic.get("sex", "adult_female_non_pregnant")
        normal_range = IndiaContext.HB_NORMAL_INDIA.get(sex, (12.0, 15.0))
        normal_mean = (normal_range[0] + normal_range[1]) / 2
        adjustment = (normal_mean - 14.0) * 0.3
        return max(4.0, min(18.0, raw_hb + adjustment))
    
    def _classify_risk_india(self, hb, demographic):
        """WHO anaemia classification for Indian context."""
        sex = demographic.get("sex", "adult_female_non_pregnant")
        
        if sex == "adult_male":
            thresholds = {"severe": 8.0, "moderate": 11.0, "mild": 12.5}
        elif sex == "adult_female_pregnant":
            thresholds = {"severe": 7.0, "moderate": 9.0, "mild": 10.5}
        else:
            thresholds = {"severe": 8.0, "moderate": 10.0, "mild": 11.5}
        
        if hb < thresholds["severe"]:
            return "Severe", "गंभीर"
        elif hb < thresholds["moderate"]:
            return "Moderate", "मध्यम"
        elif hb < thresholds["mild"]:
            return "Mild", "हल्का"
        else:
            return "Normal", "सामान्य"
    
    def _detect_confounders(self, original, corrected):
        """Detect common Indian eye conditions."""
        detected = []
        img_hsv = cv2.cvtColor((corrected * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        sclera_mask = self._simple_sclera_mask(original)
        if sclera_mask is not None and np.any(sclera_mask):
            sclera_hue = img_hsv[sclera_mask > 0][:, 0]
            if len(sclera_hue) > 0 and np.median(sclera_hue) > 25 and np.median(sclera_hue) < 45:
                detected.append("possible_jaundice")
        
        return detected
    
    def _simple_sclera_mask(self, img_bgr):
        """Quick sclera mask for confounder detection."""
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(img_lab)
        mask = (L > 170) & (np.abs(a.astype(int) - 128) < 15)
        return mask.astype(np.uint8) * 255
    
    def _should_refer(self, hb, risk_level, confounders, confidence):
        """Determine if clinical referral is recommended."""
        if risk_level == "Severe":
            return True
        if risk_level == "Moderate" and confidence < 0.6:
            return True
        if confounders:
            return True
        if hb < 10.0:
            return True
        return False
    
    def _format_ayushman_bharat(self, hb, risk, confidence):
        """Format output for Ayushman Bharat Digital Mission."""
        return {
            "resourceType": "Observation",
            "status": "preliminary",
            "valueQuantity": {
                "value": hb,
                "unit": "g/dL",
                "code": "g/dL"
            },
            "interpretation": [{"text": f"{risk} anaemia (screening)"}],
            "note": [{"text": f"Confidence: {confidence:.2f}. Screening only."}]
        }
    
    def _error_result(self, message, cal_quality):
        """Return error result when segmentation fails."""
        return IndiaAdaptedResult(
            pallor_index=0.5,
            estimated_hb=11.0,
            confidence=0.0,
            risk_level="Unknown",
            risk_level_hi="अज्ञात",
            calibration_quality=cal_quality,
            demographic_adjusted_hb=11.0,
            confounders_detected=[],
            referral_required=False,
            ayushman_bharat_format={}
        )


# ─── Clinical Study Mode ──────────────────────────────────────────────────────

class ClinicalStudyMode:
    """For IRB-approved clinical validation studies only."""
    
    def __init__(self, study_id: str, output_dir: str = "./study_data"):
        self.study_id = study_id
        self.output_dir = output_dir
        self.results = []
    
    def record_validation(self, image_hash: str, predicted_hb: float,
                          ground_truth_hb: float, demographic: Dict):
        """Record a validated prediction."""
        self.results.append({
            "image_hash": image_hash,
            "predicted_hb": predicted_hb,
            "ground_truth_hb": ground_truth_hb,
            "error": predicted_hb - ground_truth_hb,
            "absolute_error": abs(predicted_hb - ground_truth_hb),
            "demographic": demographic,
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_analysis_report(self) -> Dict:
        """Generate statistical analysis."""
        if not self.results:
            return {"error": "No validation data recorded"}
        
        errors = [r["absolute_error"] for r in self.results]
        
        return {
            "study_id": self.study_id,
            "n_samples": len(self.results),
            "mae": np.mean(errors),
            "rmse": np.sqrt(np.mean(np.square(errors))),
            "bias": np.mean([r["error"] for r in self.results]),
            "std_dev": np.std(errors)
        }


# ─── Synthetic Image Generation for Testing ──────────────────────────────────

def create_synthetic_eye_image(hb_level: float = 11.0) -> np.ndarray:
    """
    Create a synthetic eye image for testing.
    
    Args:
        hb_level: Target haemoglobin (g/dL) - affects conjunctival redness
                 8.0 = severe anaemia (pale), 15.0 = normal (red)
    """
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    
    # Skin background
    img[:, :] = [100, 140, 180]
    
    # Sclera (white of eye)
    cv2.ellipse(img, (250, 150), (120, 80), 0, 0, 360, (240, 238, 235), -1)
    
    # Iris
    cv2.circle(img, (280, 150), 35, (80, 70, 60), -1)
    cv2.circle(img, (280, 150), 15, (40, 35, 30), -1)
    
    # Conjunctiva (colour depends on Hb)
    # Normal Hb = 15 -> redness ~170, Severe anaemia = 8 -> redness ~120
    redness = int(120 + (hb_level - 8) * 7.14)  # Scale from 8g/dL to 15g/dL
    redness = max(110, min(180, redness))
    
    cv2.ellipse(img, (250, 280), (140, 50), 0, 0, 360, 
                (redness - 40, redness - 20, redness), -1)
    
    # Add noise
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def demo_with_synthetic_images():
    """Demo with multiple synthetic images showing different Hb levels."""
    
    DISCLAIMER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  RESEARCH PROTOTYPE - NOT FOR MEDICAL USE  ⚠️                             ║
║  For IRB-approved clinical studies only. Not a diagnostic device.            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(DISCLAIMER)
    print("\n" + "="*70)
    print("ANAEMIASCOPE INDIA - RESEARCH PROTOTYPE v0.2")
    print("Synthetic Image Demo (for testing only)")
    print("="*70 + "\n")
    
    scope = AnaemiaScopeIndia(language="en")
    
    # Test with different Hb levels
    test_cases = [
        ("Severe Anaemia (8.0 g/dL)", 8.0),
        ("Moderate Anaemia (9.5 g/dL)", 9.5),
        ("Mild Anaemia (11.0 g/dL)", 11.0),
        ("Normal (13.5 g/dL)", 13.5),
    ]
    
    for name, hb_target in test_cases:
        synthetic_img = create_synthetic_eye_image(hb_target)
        
        # Add demographic info
        demographic = {"age": 30, "sex": "adult_female_non_pregnant"}
        
        result = scope.analyse(synthetic_img, demographic)
        
        print(f"📋 {name}")
        print(f"   ┌─────────────────────────────────────────────────────────┐")
        print(f"   │ Pallor Index:     {result.pallor_index}                      │")
        print(f"   │ Estimated Hb:     {result.estimated_hb} g/dL (raw)          │")
        print(f"   │ Adjusted Hb:      {result.demographic_adjusted_hb} g/dL        │")
        print(f"   │ Risk Level:       {result.risk_level} ({result.risk_level_hi})  │")
        print(f"   │ Confidence:       {result.confidence:.0%}                       │")
        print(f"   │ Referral Needed:  {'YES ⚠️' if result.referral_required else 'No'}                      │")
        print(f"   │ Calibration:      {result.calibration_quality}                    │")
        print(f"   └─────────────────────────────────────────────────────────┘\n")
    
    print("="*70)
    print("⚠️  REMINDER: This is a RESEARCH PROTOTYPE only.")
    print("   Not approved for clinical diagnosis.")
    print("   Always confirm with laboratory CBC test.")
    print("="*70)


def real_image_demo(image_path: str):
    """Run analysis on a real image file."""
    
    DISCLAIMER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS ⚠️                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(DISCLAIMER)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Cannot load image from {image_path}")
        return
    
    scope = AnaemiaScopeIndia(language="en")
    demographic = {"age": 30, "sex": "adult_female_non_pregnant"}
    
    result = scope.analyse(img, demographic)
    
    print("\n" + "="*60)
    print("ANAEMIASCOPE ANALYSIS RESULT")
    print("="*60)
    print(f"\n  Pallor Index:      {result.pallor_index}")
    print(f"  Estimated Hb:      {result.estimated_hb} g/dL")
    print(f"  Demographic Adj:   {result.demographic_adjusted_hb} g/dL")
    print(f"  Risk Level:        {result.risk_level}")
    print(f"  Confidence:        {result.confidence:.0%}")
    print(f"  Calibration:       {result.calibration_quality}")
    print(f"  Referral:          {'YES - See doctor' if result.referral_required else 'No'}")
    
    if result.confounders_detected:
        print(f"  Confounders:       {', '.join(result.confounders_detected)}")
    
    print(f"\n  📄 Ayushman Bharat Format:")
    print(json.dumps(result.ayushman_bharat_format, indent=2))
    print("\n" + "="*60)
    print("⚠️  This is a screening estimate only.")
    print("   Confirm with laboratory haemoglobin test.")
    print("="*60)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run on real image
        real_image_demo(sys.argv[1])
    else:
        # Run synthetic demo
        demo_with_synthetic_images()