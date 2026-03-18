/**
 * Methylmercury multicompartment pharmacokinetic model.
 * Ported from the original Node.js implementation by Quintin Pope.
 *
 * References:
 *   Pope & Rand, Toxicological Sciences, 2021
 *   https://github.com/QuintinPope/methylmercury-website
 */

// ---------- Scaling functions ----------

function organScale(weight: number, meanWeight: number): number {
  return weight / meanWeight;
}

function flowScale(weight: number, meanWeight: number): number {
  return Math.pow(weight, 0.75) / Math.pow(meanWeight, 0.75);
}

function rateScale(weight: number, meanWeight: number): number {
  return Math.pow(weight, -0.25) / Math.pow(meanWeight, -0.25);
}

// ---------- Dosing ----------

function slowDoses(
  t: number,
  period: number,
  dose: number,
  halflife: number,
  end: number,
): number {
  if (t > end) return 0;
  const tMod = t % period;
  const lambda = Math.LN2 / halflife;
  return dose * lambda * Math.exp(-lambda * tMod);
}

// ---------- Body-type parameter tables ----------

interface BodyParams {
  meanWeight: number;
  // Volumes (L)
  VPl: number; VRBC: number; VBr: number; VKi: number; VLv: number;
  VGt: number; VGL: number; VMu: number; VFa: number; VSp: number; VRp: number;
  // Flows (L/h) — cardiac output fractions * CO * 60 * 0.6
  QPl: number; QBr: number; QKi: number; QLv: number;
  QGt: number; QMu: number; QFa: number; QSp: number; QRp: number;
}

// Partition coefficients (tissue:plasma)
const P = {
  Br: 36, Mu: 24, Rp: 12, Sp: 24,
  Lv: 60, Gt: 12, Ki: 48, Fa: 1.5,
};

// Default kinetic rate constants
export const DEFAULT_RATES = {
  KFe: 0.00661087,
  KBi: 1.71883,
  kGLI: 0.0136666,
  kEx: 0.273333,
  kLvI: 0.000911109,
  kAbs: 0.273333,
};

// body type index: 0 = child, 1 = adult male, 2 = adult female
function getBodyParams(bodyType: number, weight: number): BodyParams {
  // Reference parameters for each body type
  const tables: Record<number, {
    meanWeight: number;
    volumes: number[]; // VPl, VRBC, VBr, VKi, VLv, VGt, VGL, VMu, VFa, VSp, VRp
    CO: number; // cardiac output L/min
    flowFracs: number[]; // QBr, QKi, QLv, QGt, QMu, QFa, QSp, QRp (fractions of CO)
  }> = {
    0: { // Child
      meanWeight: 19.28,
      volumes: [0.735, 0.49, 1.31, 0.175, 0.57, 0.22, 0.08, 5.60, 3.45, 6.31, 1.97],
      CO: 3.14,
      flowFracs: [0.144, 0.176, 0.070, 0.167, 0.122, 0.052, 0.145, 0.124],
    },
    1: { // Adult male
      meanWeight: 83.12,
      volumes: [3.164, 2.109, 1.45, 0.31, 1.80, 1.03, 0.35, 29.33, 14.00, 13.42, 3.69],
      CO: 6.34,
      flowFracs: [0.114, 0.175, 0.065, 0.146, 0.175, 0.052, 0.145, 0.128],
    },
    2: { // Adult female
      meanWeight: 68.91,
      volumes: [2.584, 1.722, 1.30, 0.275, 1.40, 0.87, 0.3, 17.42, 18.70, 8.67, 3.22],
      CO: 5.32,
      flowFracs: [0.114, 0.175, 0.065, 0.146, 0.121, 0.085, 0.145, 0.149],
    },
  };

  const t = tables[bodyType] ?? tables[1];
  const mW = t.meanWeight;
  const oS = (v: number) => v * organScale(weight, mW);
  const fS = (f: number) => f * t.CO * 60 * 0.6 * flowScale(weight, mW);

  return {
    meanWeight: mW,
    VPl: oS(t.volumes[0]), VRBC: oS(t.volumes[1]), VBr: oS(t.volumes[2]),
    VKi: oS(t.volumes[3]), VLv: oS(t.volumes[4]), VGt: oS(t.volumes[5]),
    VGL: oS(t.volumes[6]), VMu: oS(t.volumes[7]), VFa: oS(t.volumes[8]),
    VSp: oS(t.volumes[9]), VRp: oS(t.volumes[10]),
    QPl: 0, // not used directly
    QBr: fS(t.flowFracs[0]), QKi: fS(t.flowFracs[1]), QLv: fS(t.flowFracs[2]),
    QGt: fS(t.flowFracs[3]), QMu: fS(t.flowFracs[4]), QFa: fS(t.flowFracs[5]),
    QSp: fS(t.flowFracs[6]), QRp: fS(t.flowFracs[7]),
  };
}

// ---------- ODE system ----------

type State = number[];
type DerivFn = (t: number, y: State) => State;

function makeDerivatives(
  bp: BodyParams,
  rates: typeof DEFAULT_RATES,
  dose: number,
  dosingEnd: number,
): DerivFn {
  const mW = bp.meanWeight;
  const kRPl = 0.3 * rateScale(bp.VPl / organScale(1, 1) * mW, mW); // simplified
  const kPlR = 3.0 * rateScale(bp.VPl / organScale(1, 1) * mW, mW);

  // Actually use simpler scaled versions matching original code
  const weight = bp.VPl / (3.164 / 83.12); // estimate weight back (approx)
  const kRPlScaled = 0.3 * rateScale(weight, mW);
  const kPlRScaled = 3.0 * rateScale(weight, mW);

  const period = 24 * 7; // weekly dosing (hours)
  const absHalflife = 3; // absorption half-life (hours)

  return (t: number, y: State): State => {
    const dy = new Array(13).fill(0);

    const cPl = y[0] / bp.VPl; // plasma concentration

    // Dosing input
    const doseRate = slowDoses(t, period, dose, absHalflife, dosingEnd);

    // Plasma (y[0])
    dy[0] =
      bp.QBr * (y[1] / (bp.VBr * P.Br) - cPl) +
      bp.QMu * (y[2] / (bp.VMu * P.Mu) - cPl) +
      bp.QFa * (y[3] / (bp.VFa * P.Fa) - cPl) +
      bp.QRp * (y[4] / (bp.VRp * P.Rp) - cPl) +
      bp.QSp * (y[5] / (bp.VSp * P.Sp) - cPl) +
      (bp.QLv + bp.QGt) * (y[6] / (bp.VLv * P.Lv) - cPl) +
      bp.QKi * (y[9] / (bp.VKi * P.Ki) - cPl) +
      kRPlScaled * y[10] - kPlRScaled * y[0];

    // Brain (y[1])
    dy[1] = bp.QBr * (cPl - y[1] / (bp.VBr * P.Br));

    // Muscle (y[2])
    dy[2] = bp.QMu * (cPl - y[2] / (bp.VMu * P.Mu));

    // Fat (y[3])
    dy[3] = bp.QFa * (cPl - y[3] / (bp.VFa * P.Fa));

    // Richly perfused (y[4])
    dy[4] = bp.QRp * (cPl - y[4] / (bp.VRp * P.Rp));

    // Slowly perfused (y[5])
    dy[5] = bp.QSp * (cPl - y[5] / (bp.VSp * P.Sp));

    // Liver (y[6])
    dy[6] =
      bp.QLv * (cPl - y[6] / (bp.VLv * P.Lv)) +
      bp.QGt * (y[7] / (bp.VGt * P.Gt) - y[6] / (bp.VLv * P.Lv)) -
      rates.KBi * y[6] / (bp.VLv * P.Lv) -
      rates.kLvI * y[6] / (bp.VLv * P.Lv);

    // Gut tissue (y[7])
    dy[7] =
      bp.QGt * (cPl - y[7] / (bp.VGt * P.Gt)) +
      rates.kAbs * y[8] -
      rates.kEx * y[7] / (bp.VGt * P.Gt);

    // Gut lumen (y[8])
    dy[8] =
      rates.KBi * y[6] / (bp.VLv * P.Lv) +
      rates.kEx * y[7] / (bp.VGt * P.Gt) -
      rates.kAbs * y[8] -
      rates.kGLI * y[8] -
      rates.KFe * y[8] +
      doseRate;

    // Kidney (y[9])
    dy[9] = bp.QKi * (cPl - y[9] / (bp.VKi * P.Ki));

    // RBC (y[10])
    dy[10] = kPlRScaled * y[0] - kRPlScaled * y[10];

    // Tracking: cumulative biliary transfer
    dy[11] = rates.KBi * y[6] / (bp.VLv * P.Lv);

    // Tracking: cumulative gut excretion
    dy[12] = rates.kEx * y[7] / (bp.VGt * P.Gt);

    return dy;
  };
}

// ---------- RK4 solver ----------

function rk4Step(f: DerivFn, t: number, y: State, dt: number): State {
  const n = y.length;
  const k1 = f(t, y);
  const y2 = y.map((v, i) => v + 0.5 * dt * k1[i]);
  const k2 = f(t + 0.5 * dt, y2);
  const y3 = y.map((v, i) => v + 0.5 * dt * k2[i]);
  const k3 = f(t + 0.5 * dt, y3);
  const y4 = y.map((v, i) => v + dt * k3[i]);
  const k4 = f(t + dt, y4);
  return y.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

// ---------- Public interface ----------

export interface SimulationInput {
  weightKg: number;
  bodyType: number; // 0=child, 1=adult male, 2=adult female
  fishPpm: number;
  mealSizeG: number; // grams
  rates?: typeof DEFAULT_RATES;
}

export interface SimulationResult {
  timeHours: number[];
  timeDays: number[];
  bloodConc: number[];  // µg/L
  brainConc: number[];  // µg/L
  muscleConc: number[]; // µg/L
  halfLifeDays: number;
  eqTimeDays: number;
}

export function runSimulation(input: SimulationInput): SimulationResult {
  const rates = input.rates ?? DEFAULT_RATES;
  const bp = getBodyParams(input.bodyType, input.weightKg);
  const dose = input.fishPpm * input.mealSizeG; // µg total MeHg per meal

  const tEnd = 1400; // hours
  const dt = 0.5;    // step size (hours)
  const sampleEvery = 6; // record every 6 steps = 3 hours
  const dosingEnd = tEnd - 200; // stop dosing 200h before end to measure decay

  const deriv = makeDerivatives(bp, rates, dose, dosingEnd);
  let y: State = new Array(13).fill(0);

  const timeHours: number[] = [];
  const bloodConc: number[] = [];
  const brainConc: number[] = [];
  const muscleConc: number[] = [];

  const VBlood = bp.VPl + bp.VRBC;

  for (let step = 0; step * dt <= tEnd; step++) {
    const t = step * dt;
    if (step % sampleEvery === 0) {
      timeHours.push(t);
      bloodConc.push((y[0] + y[10]) / VBlood);
      brainConc.push(y[1] / bp.VBr);
      muscleConc.push(y[2] / bp.VMu);
    }
    y = rk4Step(deriv, t, y, dt);
  }

  // Compute half-life from terminal decay
  const decayStart = timeHours.findIndex((t) => t >= dosingEnd + 20);
  let halfLife = 50 * 24; // default fallback (days)
  if (decayStart >= 0 && bloodConc[decayStart] > 1e-12) {
    const cStart = bloodConc[decayStart];
    const cEnd = bloodConc[bloodConc.length - 1];
    const tSpan = (timeHours[timeHours.length - 1] - timeHours[decayStart]);
    if (cEnd > 0 && cEnd < cStart) {
      const lambda = Math.log(cStart / cEnd) / tSpan;
      halfLife = Math.LN2 / lambda / 24; // convert hours to days
    }
  }

  const eqTime = halfLife * 5; // ~5 half-lives to equilibrium

  return {
    timeHours,
    timeDays: timeHours.map((t) => t / 24),
    bloodConc,
    brainConc,
    muscleConc,
    halfLifeDays: Math.round(halfLife * 10) / 10,
    eqTimeDays: Math.round(eqTime * 10) / 10,
  };
}

// Fish species with mercury concentrations (ppm)
export const FISH_SPECIES = [
  { name: "Swordfish", ppm: 0.280 },
  { name: "Shark", ppm: 0.200 },
  { name: "Tuna (Ahi)", ppm: 0.140 },
  { name: "Cod", ppm: 0.130 },
  { name: "Halibut", ppm: 0.110 },
  { name: "Salmon", ppm: 0.097 },
  { name: "Lobster", ppm: 0.093 },
  { name: "Pollock", ppm: 0.060 },
  { name: "Tilapia", ppm: 0.050 },
  { name: "Anchovy", ppm: 0.002 },
];

export const BODY_TYPES = [
  { label: "Child", value: 0 },
  { label: "Adult Male", value: 1 },
  { label: "Adult Female", value: 2 },
];
