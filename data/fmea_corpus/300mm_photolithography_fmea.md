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
  1. Wafer stage handling robotic wand abrasion and particle entrapment on chuck pins.
  2. Photoresist collapse and pattern line shearing from immersion fluid turbulence.
  3. Laser interferometer focus drift ($> 18\text{nm}$) on high-acceleration wafer stage.

### 2.2 Corrective Action SOP (Step-by-Step)
1. **Emergency Line Stop:** Intercept wafer transfer FOUP; suspend incoming lot routing to Scanner Track 2.
2. **Wafer Pre-Aligner & Stage Inspection:**
   * Inspect wafer stage handling robotic end-effector vacuum wand; replace worn suction cups (Part #AWH-WAND-PEEK-300).
   * Verify chuck pin planarity deviation $< 15\text{nm}$ across all 1,200 vacuum burls to eliminate focus drift.
3. **Immersion Hood Fluidics:**
   * Purge immersion liquid hood with ultra-pure degassed water; verify zero micro-bubble nucleation to prevent photoresist collapse.
4. **Tool Requalification:**
   * Run 2 bare monitor silicon wafers through wafer stage transport without exposure; verify zero added particles ($d > 20\text{nm}$).
