# 300mm Reactive Ion Etching (RIE) Chamber 3 FMEA SOP
**Document ID:** FMEA-SOP-ETCH-300-CH3  
**SEMI Standard:** SEMI E10-0304 Standard for Equipment Reliability and Maintainability  
**Process Tool:** Lam Research Kiyo45 / AMAT Centris 300mm Dielectric & Conductor Etcher  
**Classification Coverage:** Center, Donut, Edge-Ring, Short, Open_circuit, Spur  
**Target Station:** Chamber 3 (Dual-Frequency Inductively Coupled Plasma / ICP Gate Etch)  
**Revision:** v2.4.0 (Approved for 3nm/5nm Cleanroom Metrology)  

---

## 1. Executive Summary & Process Physics
During 300mm wafer dielectric and metal gate plasma etching, energetic ions ($CF_4, CHF_3, Cl_2, HBr, O_2$) are accelerated across the plasma sheath towards the electrostatic chuck (ESC). Yield excursions in this chamber present distinct spatial radial signatures on wafer-bin maps (WBM) correlated with micro-die conductor bridging or trench voids.

---

## 2. Center Radial Excursion & Micro-Bridging Shorts
### 2.1 Failure Mechanism & Physical Root Cause
* **Wafer Spatial Pattern:** Center cluster ($r < 45\text{mm}$, radially symmetric defect density $D_0 > 14.2\text{ defects/cm}^2$).
* **Die Micro-Defect Code:** `Short` / Incomplete dielectric trench clearing and metal stringer bridging ($<18\text{nm}$ spacing).
* **Physical Root Cause:**
  1. Inner-zone RF match capacitor ($C_{\text{tune}}$) impedance drift causing localized center plasma density peaking ($n_e > 2.8 \times 10^{11}\text{ cm}^{-3}$).
  2. Thermal runaway due to center-zone Helium backside cooling pressure drop below threshold ($P_{\text{He, center}} < 7.4\text{ Torr}$, nominal $10.0 \pm 0.5\text{ Torr}$) from silicone polymer seal degradation.

### 2.2 Corrective Action SOP (Step-by-Step)
1. **Immediate State Change:** Trigger automated tool lock via GEM/SECS interface; set Chamber 3 to `UNSCHEDULED_DOWNTIME_MAINTENANCE`.
2. **Backside Helium Leak Check:**
   * Evacuate loadlock; apply $15\text{ Torr}$ static He test pressure.
   * Measure leak rate into high-vacuum chamber over $120\text{s}$. If leak rate $> 0.045\text{ sccm}$, replace ESC ceramic puck seal ring (Part #ESC-300-SEAL-09).
3. **RF Match Network Tuning:**
   * Connect high-voltage vector impedance analyzer to the $13.56\text{ MHz}$ source and $2\text{ MHz}$ bias RF generators.
   * Recalibrate center/edge power splitting coil ratio to $0.48 : 0.52$.
4. **Plasma Chamber De-Polymerization Clean:**
   * Execute $25\text{ min}$ automated recipe `WAC_O2_AR_HIGH_POWER` ($800\text{W}$ source power, $120\text{ sccm } O_2$, $40\text{ sccm } Ar$) to strip fluorocarbon polymer buildup from the quartz showerhead gas injection nozzles.
5. **Metrology Qualification Run:**
   * Process 3 blanket $100\text{nm}$ $SiO_2$ monitor wafers.
   * Verify etch rate uniformity $3\sigma < 1.45\%$ using 49-point ellipsometry.

---

## 3. Donut & Edge-Ring Radial Excursions
### 3.1 Failure Mechanism & Physical Root Cause
* **Wafer Spatial Pattern:** Donut annular band ($55\text{mm} < r < 115\text{mm}$) or Edge-Ring ($r > 138\text{mm}$, outer $3\text{mm}$ exclusion zone).
* **Die Micro-Defect Code:** `Mouse_bite` / `Spur` (micro-notching on polysilicon sidewalls).
* **Physical Root Cause:**
  1. Quartz/Yttria focus ring physical erosion step height exceeding $0.75\text{mm}$ (spec: $<0.50\text{mm}$), resulting in plasma sheath bending at the wafer bevel edge and micro-arcing.
  2. Outer gas injection ring flow ratio imbalance ($C_4F_8 / O_2 = 1.62$, nominal $1.20 \pm 0.05$).

### 3.2 Corrective Action SOP (Step-by-Step)
1. Vent chamber with heated $N_2$; measure focus ring step profile using portable optical chromatic confocal laser sensor.
2. Replace worn focus ring assembly with pre-conditioned Yttria-coated ceramic ring (Part #FR-Y2O3-300).
3. Check MFC (Mass Flow Controller) calibration for outer gas feed channels #4 and #5.
4. Execute chamber season recipe (5 dummy silicon wafers under standard etch conditions).
5. Inspect edge die yield on lot rerun; confirm edge yield $> 98.1\%$.
