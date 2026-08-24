# 300mm Extreme Ultraviolet (EUV) & Immersion Lithography FMEA SOP
**Document ID:** FMEA-SOP-LITHO-300-SC2  
**SEMI Standard:** SEMI E10-0304 Standard for Equipment Reliability and Maintainability  
**Process Tool:** ASML TWINSCAN NXE:3400B EUV / Immersion ArFi Scanner  
**Classification Coverage:** Scratch, Loc, Missing_hole, Mouse_bite, Open_circuit  
**Target Station:** Scanner Track 2 (Wafer Stage 2 & Reticle Pellicle Chamber)  
**Revision:** v3.1.0 (Approved for 3nm/5nm Cleanroom Metrology)  

---

## 1. Executive Summary & Optical Physics
Lithography exposure transfers sub-20nm circuit patterns from quartz reticles to photoresist-coated wafers. Physical defects on reticles, laser focus aberrations, immersion fluid micro-bubbles, or robotic handling scratches create localized catastrophic circuit open failures.

---

## 2. Linear & Curvilinear Scratch Excursions
### 2.1 Failure Mechanism & Physical Root Cause
* **Wafer Spatial Pattern:** Linear or multi-arc `Scratch` pattern traversing multiple die rows ($L > 40\text{mm}$).
* **Die Micro-Defect Code:** `Open_circuit` / Severed copper interconnect lines ($w < 22\text{nm}$) and sheared contact pads.
* **Physical Root Cause:**
  1. Robotic atmospheric wafer handler (AWH) end-effector vacuum wand abrasion due to worn PEEK suction cups.
  2. Particle entrapment ($d > 3.5\mu\text{m}$) on high-acceleration wafer pre-aligner chuck pins ($a > 18\text{ m/s}^2$).

### 2.2 Corrective Action SOP (Step-by-Step)
1. **Emergency Line Stop:** Intercept wafer transfer FOUP; suspend incoming lot routing to Scanner Track 2.
2. **Robotic End-Effector Inspection:**
   * Inspect robot arm PEEK contact pads under $50\times$ optical boroscope for micro-fractures or embedded silicon debris.
   * Replace vacuum end-effector assembly (Part #AWH-WAND-PEEK-300).
3. **Chuck Pin Ultrasonic Vacuum Purge:**
   * Run automated chuck cleaning sequence: apply high-purity IPA wash followed by $500\text{ kPa}$ pulsed $N_2$ blowdown.
   * Verify chuck pin planarity deviation $< 15\text{nm}$ across all 1,200 vacuum burls.
4. **Particle Qualification Check:**
   * Run 2 bare monitor silicon wafers through wafer stage transport without exposure.
   * Scan wafer surface on optical defect inspection tool (KLA 2935); confirm added particle count $\Delta P = 0$ at $>20\text{nm}$ sensitivity.

---

## 3. Localized (Loc) Cluster & Missing Contact Vias
### 3.1 Failure Mechanism & Physical Root Cause
* **Wafer Spatial Pattern:** Dense localized (`Loc`) cluster localized to a specific die quadrant ($A < 12\text{mm}^2$).
* **Die Micro-Defect Code:** `Missing_hole` / Contact via unexposed or partially printed in resist.
* **Physical Root Cause:**
  1. Immersion fluid degasser membrane saturation leading to micro-bubble pinhole formation ($d < 80\text{nm}$) in ultra-pure water (UPW) layer during immersion scanning ($NA = 1.35$).
  2. Foreign organic particle flake on reticle pellicle membrane at Reticle Slot 2.

### 3.2 Corrective Action SOP (Step-by-Step)
1. Inspect reticle pellicle in-situ using scanner transmission defect camera; perform reticle surface laser clean if particle detected.
2. Verify immersion UPW degasser dissolved oxygen concentration (nominal: $< 1.5\text{ ppb}$).
3. Replace POU UPW sub-10nm hollow-fiber nanofilter (Part #UPW-NF-010).
4. Run focus-exposure matrix (FEM) test wafer; confirm critical dimension uniformity ($3\sigma_{\text{CDU}} < 0.65\text{nm}$).
