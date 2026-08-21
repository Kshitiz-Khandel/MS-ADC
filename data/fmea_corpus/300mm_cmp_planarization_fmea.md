# 300mm Chemical Mechanical Planarization (CMP) FMEA SOP
**Document ID:** FMEA-SOP-CMP-300-PL1  
**SEMI Standard:** SEMI E10-0304 Standard for Equipment Reliability and Maintainability  
**Process Tool:** Applied Materials Reflexion LK 300mm CMP Polisher  
**Classification Coverage:** Edge-Loc, Random, Spurious_copper, Spur, Scratch  
**Target Station:** Platen 1 (Copper Bulk & Barrier CMP Polishing Station)  
**Revision:** v2.8.0 (Approved for 3nm/5nm Cleanroom Metrology)  

---

## 1. Executive Summary & Tribological Physics
CMP achieves global wafer surface planarization via abrasive slurry chemical reactions and mechanical polyurethane pad shear. Excursions in slurry delivery, conditioning diamond loss, or retaining ring pressure loss create edge defectivity and spurious copper bridging shorts across dies.

---

## 2. Edge-Loc Excursion & Spurious Copper Flakes
### 2.1 Failure Mechanism & Physical Root Cause
* **Wafer Spatial Pattern:** `Edge-Loc` (high defect density clustered exclusively within $6\text{mm}$ of wafer outer edge).
* **Die Micro-Defect Code:** `Spurious_copper` / Residual unpolished copper flakes causing multi-line short circuits.
* **Physical Root Cause:**
  1. Polishing head retaining ring pneumatic chamber pressure drop ($P_{\text{RR}} < 3.6\text{ psi}$, nominal $5.2 \pm 0.2\text{ psi}$), leading to wafer edge slurry cavitation and hydroplaning.
  2. Uneven polyurethane polishing pad glazing along outer radius due to diamond conditioner sweep profile calibration drift.

### 2.2 Corrective Action SOP (Step-by-Step)
1. **Tool State Isolation:** Pause Platen 1 polishing; reroute active semiconductor lots to Platen 2 backup.
2. **Retaining Ring Wear Profilometry:**
   * Measure retaining ring slot depth across 8 radial azimuths using digital micrometer.
   * If wear profile delta $> 0.35\text{mm}$ or slot depth $< 1.1\text{mm}$, replace retaining ring (Part #CMP-RR-PPS-300).
3. **Conditioner Diamond Disc Audit:**
   * Inspect diamond conditioner disk under optical microscope; verify zero diamond micro-fracture or pull-out.
   * Recalibrate conditioner arm sweep rate to $19.5\text{ sweeps/min}$ with sinusoidal profile.
4. **Slurry Delivery System Flush:**
   * Replace slurry loop point-of-use (POU) $0.45\mu\text{m}$ depth filter.
   * Flush copper slurry delivery lines with deionized water; verify pH is $7.2 \pm 0.1$ and specific gravity is $1.085$.
5. **Post-CMP Brush Clean & Lot Disposition:**
   * Calibrate PVA brush box megasonic cleaning frequency ($1.0\text{ MHz}$) and $NH_4OH$ chemistry ratio ($2.0\% \pm 0.1\%$).
   * Rework affected wafer lot through a brief $12\text{s}$ touch-up barrier polish cycle.
