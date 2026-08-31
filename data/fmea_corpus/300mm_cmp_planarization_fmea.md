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
  1. Retaining ring pressure pneumatic chamber drop ($P_{\text{RR}} < 3.6\text{ psi}$) leading to wafer edge hydroplaning.
  2. Slurry contamination and agglomeration in delivery loop causing micro-scratches and copper residue.
  3. Platen speed oscillation and diamond conditioner sweep profile calibration drift.

### 2.2 Corrective Action SOP (Step-by-Step)
1. **Tool State Isolation:** Pause Platen 1 polishing; reroute active semiconductor lots to Platen 2 backup.
2. **Retaining Ring Wear Profilometry:**
   * Adjust retaining ring pressure pneumatic chamber to nominal $5.2 \pm 0.2\text{ psi}$.
   * Measure retaining ring slot depth across 8 radial azimuths; if wear $> 0.35\text{mm}$, replace retaining ring (Part #CMP-RR-PPS-300).
3. **Slurry Delivery System Flush:**
   * Flush copper slurry delivery loop to remove slurry contamination; replace POU $0.45\mu\text{m}$ depth filter.
4. **Conditioner & Platen Calibration:**
   * Recalibrate platen speed ($P_{\text{RPM}} = 85 \pm 1\text{ RPM}$) and conditioner diamond sweep rate to $19.5\text{ sweeps/min}$.
