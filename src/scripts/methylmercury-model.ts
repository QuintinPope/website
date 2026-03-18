/**
 * Paper-faithful methylmercury PBPK model.
 *
 * Reconstructed from:
 *   Pope & Rand, Toxicological Sciences (2021)
 *   "Variation in Methylmercury Metabolism and Elimination in Humans"
 *
 * Design goals for this version:
 *   - Use the Table 1 organ volumes, blood flows, partition coefficients, and kinetic constants.
 *   - Use the paper's routing assumptions:
 *       * plasma-only perfusion with separate RBC exchange,
 *       * portal gut -> liver flow,
 *       * zero-order dose input to gut lumen,
 *       * hair, fecal, gut-lumen demethylation, and liver demethylation bookkeeping,
 *       * biliary clearance scaled from KBi [L/h/g liver] * liver size.
 *   - Preserve full mass accounting for MeHg in compartments plus eliminated / biotransformed Hg.
 *
 * State vector (17 ODEs):
 *   0  plasma MeHg amount
 *   1  brain MeHg amount
 *   2  muscle MeHg amount
 *   3  fat MeHg amount
 *   4  richly perfused tissue MeHg amount
 *   5  slowly perfused tissue MeHg amount
 *   6  liver MeHg amount
 *   7  gut tissue MeHg amount
 *   8  gut lumen MeHg amount
 *   9  kidney MeHg amount
 *   10 RBC MeHg amount
 *   11 cumulative biliary transfer to gut lumen
 *   12 cumulative gut-tissue -> gut-lumen transfer
 *   13 cumulative gut-lumen demethylation (iHg formed)
 *   14 cumulative liver demethylation (iHg formed)
 *   15 cumulative fecal MeHg excretion
 *   16 cumulative hair MeHg excretion
 */

// ---------- constants ----------

const HOURS_PER_DAY = 24;
const HOURS_PER_WEEK = 7 * HOURS_PER_DAY;
const PAPER_DOSE_DURATION_HOURS = 1; // zero-order meal input to gut lumen
const REPEATED_DOSE_WEEKS = 52;
const REPEATED_SIM_HOURS = REPEATED_DOSE_WEEKS * HOURS_PER_WEEK; // 364 days
const SINGLE_DOSE_SIM_HOURS = 50 * HOURS_PER_DAY;
const DT_HOURS = 0.5; // stable and accurate with trapezoidal stepping for this linear system
const SAMPLE_EVERY_STEPS = 6; // 3-hour chart spacing
const STATE_COUNT = 17;

const enum S {
  Pl = 0,
  Br = 1,
  Mu = 2,
  Fa = 3,
  Rp = 4,
  Sp = 5,
  Lv = 6,
  Gt = 7,
  GL = 8,
  Ki = 9,
  RBC = 10,
  CumBile = 11,
  CumGutEx = 12,
  CumGutI = 13,
  CumLivI = 14,
  CumFec = 15,
  CumHair = 16,
}

// ---------- scaling ----------

function organScale(weightKg: number, meanWeightKg: number): number {
  return weightKg / meanWeightKg;
}

function flowScale(weightKg: number, meanWeightKg: number): number {
  return Math.pow(weightKg / meanWeightKg, 0.75);
}

function rateScale(weightKg: number, meanWeightKg: number): number {
  return Math.pow(weightKg / meanWeightKg, -0.25);
}

// ---------- types ----------

export interface KineticParameters {
  kAbs: number;
  kEx: number;
  kGLI: number;
  kLvI: number;
  KBi: number;   // L / h / g liver tissue
  KFe: number;   // L / h
  KHa: number;   // L / h
  kPlR: number;  // h^-1
  kRPl: number;  // h^-1
}

interface PartitionCoefficients {
  Br: number;
  Mu: number;
  Rp: number;
  Sp: number;
  Lv: number;
  Gt: number;
  Ki: number;
  Fa: number;
  Ha: number;
}

interface BodyReference {
  meanWeightKg: number;
  volumesL: {
    VPl: number;
    VRBC: number;
    VBr: number;
    VMu: number;
    VRp: number;
    VSp: number;
    VLv: number;
    VGt: number;
    VGL: number;
    VKi: number;
    VFa: number;
  };
  bloodFlowsLMin: {
    QBr: number;
    QMu: number;
    QRp: number;
    QSp: number;
    QLv: number;
    QGt: number;
    QKi: number;
    QFa: number;
  };
  bodyTypeRates: {
    KFe: number;
    KHa: number;
  };
}

interface ModelParams {
  weightKg: number;
  meanWeightKg: number;
  VPl: number;
  VRBC: number;
  VBr: number;
  VMu: number;
  VRp: number;
  VSp: number;
  VLv: number;
  VGt: number;
  VGL: number;
  VKi: number;
  VFa: number;
  QBr: number; // plasma flow, L/h
  QMu: number;
  QRp: number;
  QSp: number;
  QLv: number;
  QGt: number;
  QKi: number;
  QFa: number;
  KBiTotal: number; // L/h total biliary clearance for the current liver size
  KFe: number;      // L/h, scaled to body weight
  KHa: number;      // L/h, scaled to body weight
  kAbs: number;     // h^-1, scaled to body weight
  kEx: number;
  kGLI: number;
  kLvI: number;
  kPlR: number;
  kRPl: number;
}

interface LinearSystem {
  A: number[][];
  inputVector: number[];
  params: ModelParams;
}

interface Trajectory {
  timeHours: number[];
  timeDays: number[];
  bloodConc: number[];
  brainConc: number[];
  muscleConc: number[];
  hairConc: number[];
  sampledTimeHours: number[];
  sampledTimeDays: number[];
  sampledBloodConc: number[];
  sampledBrainConc: number[];
  sampledMuscleConc: number[];
  sampledHairConc: number[];
  sampledMassBalanceErrorUg: number[];
  doseTotalUg: number;
  finalState: number[];
  maxAbsMassBalanceErrorUg: number;
  finalMassBalanceErrorUg: number;
}

export interface SimulationInput {
  weightKg: number;
  bodyType: number; // 0 = child, 1 = adult male, 2 = adult female
  fishPpm: number;
  mealSizeG: number; // grams, once per week
  rates?: Partial<KineticParameters>;
}

export interface EliminationFractions {
  gutDemethylationPctDose: number;
  hairPctDose: number;
  liverDemethylationPctDose: number;
  fecesPctDose: number;
  totalEliminatedPctDose: number;
  gutDemethylationPctEliminated: number;
  hairPctEliminated: number;
  liverDemethylationPctEliminated: number;
  fecesPctEliminated: number;
}

export interface SimulationResult {
  // repeated weekly dose trajectory, used for the main chart
  timeHours: number[];
  timeDays: number[];
  bloodConc: number[];   // ug/L whole blood
  brainConc: number[];   // ug/L tissue
  muscleConc: number[];  // ug/L tissue
  hairConc: number[];    // ppm, following the paper's PHa * plasma concentration output

  // summary metrics
  halfLifeDays: number;              // single-dose t1/2 from 200 h -> 300 h, paper-style
  eqTimeDays: number;                // time to 95% of final weekly-average blood level
  peakBloodConc: number;             // single-dose peak whole-blood concentration, ug/L
  steadyStateBloodConc: number;      // final weekly-average blood concentration, ug/L
  elimination50d: EliminationFractions;
  maxAbsMassBalanceErrorUg: number;
  finalMassBalanceErrorUg: number;
}

// ---------- paper parameterization ----------

const PARTITION: PartitionCoefficients = {
  Br: 30,
  Mu: 20,
  Rp: 10,
  Sp: 20,
  Lv: 50,
  Gt: 10,
  Ki: 40,
  Fa: 1.5,
  Ha: 3000,
};

const BODY_REFERENCE: Record<number, BodyReference> = {
  0: {
    meanWeightKg: 19.0,
    volumesL: {
      VPl: 0.97,
      VRBC: 0.49,
      VBr: 1.39,
      VMu: 4.90,
      VRp: 1.00,
      VSp: 0.70,
      VLv: 0.50,
      VGt: 0.28,
      VGL: 0.30,
      VKi: 0.09,
      VFa: 3.60,
    },
    bloodFlowsLMin: {
      QBr: 0.72,
      QMu: 0.15,
      QRp: 0.77,
      QSp: 0.02,
      QLv: 0.46,
      QGt: 0.22,
      QKi: 0.33,
      QFa: 0.11,
    },
    bodyTypeRates: {
      KFe: 0.00208,
      KHa: 2.3e-5,
    },
  },
  1: {
    meanWeightKg: 73.0,
    volumesL: {
      VPl: 3.46,
      VRBC: 2.36,
      VBr: 1.34,
      VMu: 32.0,
      VRp: 3.72,
      VSp: 4.00,
      VLv: 1.57,
      VGt: 1.23,
      VGL: 0.90,
      VKi: 0.32,
      VFa: 14.60,
    },
    bloodFlowsLMin: {
      QBr: 0.68,
      QMu: 0.95,
      QRp: 2.60,
      QSp: 0.12,
      QLv: 1.32,
      QGt: 0.93,
      QKi: 1.17,
      QFa: 0.29,
    },
    bodyTypeRates: {
      KFe: 0.00625,
      KHa: 4.0e-5,
    },
  },
  2: {
    meanWeightKg: 60.0,
    volumesL: {
      VPl: 2.68,
      VRBC: 1.46,
      VBr: 1.20,
      VMu: 21.0,
      VRp: 2.87,
      VSp: 4.00,
      VLv: 1.35,
      VGt: 1.17,
      VGL: 0.83,
      VKi: 0.26,
      VFa: 18.0,
    },
    bloodFlowsLMin: {
      QBr: 0.63,
      QMu: 0.63,
      QRp: 2.21,
      QSp: 0.12,
      QLv: 1.35,
      QGt: 0.91,
      QKi: 0.85,
      QFa: 0.54,
    },
    bodyTypeRates: {
      KFe: 0.00500,
      KHa: 3.4e-5,
    },
  },
};

const SHARED_REFERENCE_RATES = {
  kAbs: 0.3,
  kEx: 0.1,
  kGLI: 0.08,
  kLvI: 0.001,
  KBi: 0.00035,
  kPlR: 3.0,
  kRPl: 0.3,
};

export function getDefaultRatesForBodyType(bodyType: number): KineticParameters {
  const reference = BODY_REFERENCE[bodyType] ?? BODY_REFERENCE[1];
  return {
    ...SHARED_REFERENCE_RATES,
    KFe: reference.bodyTypeRates.KFe,
    KHa: reference.bodyTypeRates.KHa,
  };
}

export const DEFAULT_RATES = getDefaultRatesForBodyType(1);

// ---------- fish and body type options ----------

export const FISH_SPECIES = [
  { name: 'Swordfish', ppm: 0.280 },
  { name: 'Shark', ppm: 0.200 },
  { name: 'Tuna (Ahi)', ppm: 0.140 },
  { name: 'Cod', ppm: 0.130 },
  { name: 'Halibut', ppm: 0.110 },
  { name: 'Salmon', ppm: 0.097 },
  { name: 'Lobster', ppm: 0.093 },
  { name: 'Pollock', ppm: 0.060 },
  { name: 'Tilapia', ppm: 0.050 },
  { name: 'Anchovy', ppm: 0.002 },
];

export const BODY_TYPES = [
  { label: 'Child', value: 0 },
  { label: 'Adult Male', value: 1 },
  { label: 'Adult Female', value: 2 },
];

// ---------- linear algebra helpers ----------

function zerosMatrix(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

function identityMatrix(size: number): number[][] {
  const out = zerosMatrix(size, size);
  for (let i = 0; i < size; i += 1) out[i][i] = 1;
  return out;
}

function cloneMatrix(matrix: number[][]): number[][] {
  return matrix.map((row) => [...row]);
}

function addMatrices(a: number[][], b: number[][]): number[][] {
  const out = zerosMatrix(a.length, a[0].length);
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < a[i].length; j += 1) {
      out[i][j] = a[i][j] + b[i][j];
    }
  }
  return out;
}

function scaleMatrix(matrix: number[][], scalar: number): number[][] {
  const out = zerosMatrix(matrix.length, matrix[0].length);
  for (let i = 0; i < matrix.length; i += 1) {
    for (let j = 0; j < matrix[i].length; j += 1) {
      out[i][j] = matrix[i][j] * scalar;
    }
  }
  return out;
}

function multiplyMatrices(a: number[][], b: number[][]): number[][] {
  const out = zerosMatrix(a.length, b[0].length);
  for (let i = 0; i < a.length; i += 1) {
    for (let k = 0; k < b.length; k += 1) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < b[k].length; j += 1) {
        out[i][j] += aik * b[k][j];
      }
    }
  }
  return out;
}

function multiplyMatrixVector(matrix: number[][], vector: number[]): number[] {
  const out = new Array(matrix.length).fill(0);
  for (let i = 0; i < matrix.length; i += 1) {
    let sum = 0;
    for (let j = 0; j < vector.length; j += 1) {
      sum += matrix[i][j] * vector[j];
    }
    out[i] = sum;
  }
  return out;
}

function addVectors(a: number[], b: number[]): number[] {
  return a.map((value, index) => value + b[index]);
}

function scaleVector(vector: number[], scalar: number): number[] {
  return vector.map((value) => value * scalar);
}

function invertMatrix(matrix: number[][]): number[][] {
  const n = matrix.length;
  const augmented = zerosMatrix(n, 2 * n);

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) augmented[i][j] = matrix[i][j];
    augmented[i][n + i] = 1;
  }

  for (let col = 0; col < n; col += 1) {
    let pivotRow = col;
    let pivotAbs = Math.abs(augmented[col][col]);
    for (let row = col + 1; row < n; row += 1) {
      const candidate = Math.abs(augmented[row][col]);
      if (candidate > pivotAbs) {
        pivotAbs = candidate;
        pivotRow = row;
      }
    }

    if (pivotAbs < 1e-14) {
      throw new Error('Matrix inversion failed: singular system matrix.');
    }

    if (pivotRow !== col) {
      const tmp = augmented[col];
      augmented[col] = augmented[pivotRow];
      augmented[pivotRow] = tmp;
    }

    const pivot = augmented[col][col];
    for (let j = 0; j < 2 * n; j += 1) augmented[col][j] /= pivot;

    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = augmented[row][col];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j += 1) {
        augmented[row][j] -= factor * augmented[col][j];
      }
    }
  }

  const inverse = zerosMatrix(n, n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) inverse[i][j] = augmented[i][n + j];
  }
  return inverse;
}

// ---------- model construction ----------

function buildModelParams(input: SimulationInput): ModelParams {
  const reference = BODY_REFERENCE[input.bodyType] ?? BODY_REFERENCE[1];
  const defaults = getDefaultRatesForBodyType(input.bodyType);
  const userRates: KineticParameters = {
    ...defaults,
    ...(input.rates ?? {}),
  };

  const weightKg = Math.max(0.1, input.weightKg);
  const meanWeightKg = reference.meanWeightKg;

  const organSF = organScale(weightKg, meanWeightKg);
  const flowSF = flowScale(weightKg, meanWeightKg);
  const rateSF = rateScale(weightKg, meanWeightKg);

  const VPl = reference.volumesL.VPl * organSF;
  const VRBC = reference.volumesL.VRBC * organSF;
  const bloodVolume = VPl + VRBC;
  const plasmaFraction = VPl / bloodVolume;

  const scaleBloodFlowToPlasma = (flowLMin: number): number => flowLMin * 60 * flowSF * plasmaFraction;

  const VLv = reference.volumesL.VLv * organSF;

  return {
    weightKg,
    meanWeightKg,
    VPl,
    VRBC,
    VBr: reference.volumesL.VBr * organSF,
    VMu: reference.volumesL.VMu * organSF,
    VRp: reference.volumesL.VRp * organSF,
    VSp: reference.volumesL.VSp * organSF,
    VLv,
    VGt: reference.volumesL.VGt * organSF,
    VGL: reference.volumesL.VGL * organSF,
    VKi: reference.volumesL.VKi * organSF,
    VFa: reference.volumesL.VFa * organSF,
    QBr: scaleBloodFlowToPlasma(reference.bloodFlowsLMin.QBr),
    QMu: scaleBloodFlowToPlasma(reference.bloodFlowsLMin.QMu),
    QRp: scaleBloodFlowToPlasma(reference.bloodFlowsLMin.QRp),
    QSp: scaleBloodFlowToPlasma(reference.bloodFlowsLMin.QSp),
    QLv: scaleBloodFlowToPlasma(reference.bloodFlowsLMin.QLv),
    QGt: scaleBloodFlowToPlasma(reference.bloodFlowsLMin.QGt),
    QKi: scaleBloodFlowToPlasma(reference.bloodFlowsLMin.QKi),
    QFa: scaleBloodFlowToPlasma(reference.bloodFlowsLMin.QFa),
    // KBi is a per-gram liver clearance in the paper. Scaling liver size already changes total biliary clearance.
    KBiTotal: userRates.KBi * VLv * 1000,
    // Flow-like clearances scale with body mass^0.75.
    KFe: userRates.KFe * flowSF,
    KHa: userRates.KHa * flowSF,
    // First-order kinetic rates scale with body mass^-0.25.
    kAbs: userRates.kAbs * rateSF,
    kEx: userRates.kEx * rateSF,
    kGLI: userRates.kGLI * rateSF,
    kLvI: userRates.kLvI * rateSF,
    kPlR: userRates.kPlR * rateSF,
    kRPl: userRates.kRPl * rateSF,
  };
}

function buildLinearSystem(input: SimulationInput): LinearSystem {
  const p = buildModelParams(input);
  const A = zerosMatrix(STATE_COUNT, STATE_COUNT);
  const inputVector = new Array(STATE_COUNT).fill(0);
  inputVector[S.GL] = 1;

  const totalPlasmaOutflow = p.QBr + p.QMu + p.QFa + p.QRp + p.QSp + p.QKi + p.QLv + p.QGt;

  // Plasma
  A[S.Pl][S.Pl] += -totalPlasmaOutflow / p.VPl;
  A[S.Pl][S.Pl] += -p.kPlR;
  A[S.Pl][S.Pl] += -(p.KHa * PARTITION.Ha) / p.VPl;
  A[S.Pl][S.Br] += p.QBr / (p.VBr * PARTITION.Br);
  A[S.Pl][S.Mu] += p.QMu / (p.VMu * PARTITION.Mu);
  A[S.Pl][S.Fa] += p.QFa / (p.VFa * PARTITION.Fa);
  A[S.Pl][S.Rp] += p.QRp / (p.VRp * PARTITION.Rp);
  A[S.Pl][S.Sp] += p.QSp / (p.VSp * PARTITION.Sp);
  A[S.Pl][S.Lv] += (p.QLv + p.QGt) / (p.VLv * PARTITION.Lv);
  A[S.Pl][S.Ki] += p.QKi / (p.VKi * PARTITION.Ki);
  A[S.Pl][S.RBC] += p.kRPl;

  // Brain
  A[S.Br][S.Pl] += p.QBr / p.VPl;
  A[S.Br][S.Br] += -p.QBr / (p.VBr * PARTITION.Br);

  // Muscle
  A[S.Mu][S.Pl] += p.QMu / p.VPl;
  A[S.Mu][S.Mu] += -p.QMu / (p.VMu * PARTITION.Mu);

  // Fat
  A[S.Fa][S.Pl] += p.QFa / p.VPl;
  A[S.Fa][S.Fa] += -p.QFa / (p.VFa * PARTITION.Fa);

  // Richly perfused tissue
  A[S.Rp][S.Pl] += p.QRp / p.VPl;
  A[S.Rp][S.Rp] += -p.QRp / (p.VRp * PARTITION.Rp);

  // Slowly perfused tissue
  A[S.Sp][S.Pl] += p.QSp / p.VPl;
  A[S.Sp][S.Sp] += -p.QSp / (p.VSp * PARTITION.Sp);

  // Liver
  A[S.Lv][S.Pl] += p.QLv / p.VPl;
  A[S.Lv][S.Pl] += -p.KBiTotal / p.VPl;
  A[S.Lv][S.Lv] += -p.QLv / (p.VLv * PARTITION.Lv);
  A[S.Lv][S.Lv] += -p.QGt / (p.VLv * PARTITION.Lv);
  A[S.Lv][S.Lv] += -p.kLvI;
  A[S.Lv][S.Gt] += p.QGt / (p.VGt * PARTITION.Gt);

  // Gut tissue
  A[S.Gt][S.Pl] += p.QGt / p.VPl;
  A[S.Gt][S.Gt] += -p.QGt / (p.VGt * PARTITION.Gt);
  A[S.Gt][S.Gt] += -p.kEx;
  A[S.Gt][S.GL] += p.kAbs;

  // Gut lumen
  A[S.GL][S.Pl] += p.KBiTotal / p.VPl;
  A[S.GL][S.Gt] += p.kEx;
  A[S.GL][S.GL] += -p.kAbs;
  A[S.GL][S.GL] += -p.kGLI;
  A[S.GL][S.GL] += -p.KFe / p.VGL;

  // Kidney
  A[S.Ki][S.Pl] += p.QKi / p.VPl;
  A[S.Ki][S.Ki] += -p.QKi / (p.VKi * PARTITION.Ki);

  // RBCs
  A[S.RBC][S.Pl] += p.kPlR;
  A[S.RBC][S.RBC] += -p.kRPl;

  // Cumulative bookkeeping states
  A[S.CumBile][S.Pl] += p.KBiTotal / p.VPl;
  A[S.CumGutEx][S.Gt] += p.kEx;
  A[S.CumGutI][S.GL] += p.kGLI;
  A[S.CumLivI][S.Lv] += p.kLvI;
  A[S.CumFec][S.GL] += p.KFe / p.VGL;
  A[S.CumHair][S.Pl] += (p.KHa * PARTITION.Ha) / p.VPl;

  return { A, inputVector, params: p };
}

// ---------- integration ----------

function buildTrapezoidStepper(system: LinearSystem, dtHours: number): {
  step: (state: number[], doseRateUgPerHour: number) => number[];
} {
  const I = identityMatrix(STATE_COUNT);
  const left = addMatrices(I, scaleMatrix(system.A, -0.5 * dtHours));
  const right = addMatrices(I, scaleMatrix(system.A, 0.5 * dtHours));
  const leftInv = invertMatrix(left);
  const transition = multiplyMatrices(leftInv, right);
  const inputTerm = multiplyMatrixVector(leftInv, scaleVector(system.inputVector, dtHours));

  return {
    step: (state: number[], doseRateUgPerHour: number): number[] => {
      const homogeneous = multiplyMatrixVector(transition, state);
      const forced = scaleVector(inputTerm, doseRateUgPerHour);
      return addVectors(homogeneous, forced);
    },
  };
}

function uniformIntegral(values: number[], startIndex: number, endIndex: number, dtHours: number): number {
  let area = 0;
  for (let i = startIndex; i < endIndex; i += 1) {
    area += 0.5 * dtHours * (values[i] + values[i + 1]);
  }
  return area;
}

function findClosestIndex(timeHours: number[], targetHours: number): number {
  const raw = Math.round(targetHours / DT_HOURS);
  return Math.max(0, Math.min(timeHours.length - 1, raw));
}

function bloodConcUgPerL(state: number[], p: ModelParams): number {
  const bloodVolume = p.VPl + p.VRBC;
  return (state[S.Pl] + state[S.RBC]) / bloodVolume;
}

function brainConcUgPerL(state: number[], p: ModelParams): number {
  return state[S.Br] / p.VBr;
}

function muscleConcUgPerL(state: number[], p: ModelParams): number {
  return state[S.Mu] / p.VMu;
}

function hairConcPpm(state: number[], p: ModelParams): number {
  return PARTITION.Ha * (state[S.Pl] / p.VPl);
}

function systemMercuryUg(state: number[]): number {
  // Cumulative bile and gut-tissue->lumen are internal transfers, not externalized mass.
  return (
    state[S.Pl] +
    state[S.Br] +
    state[S.Mu] +
    state[S.Fa] +
    state[S.Rp] +
    state[S.Sp] +
    state[S.Lv] +
    state[S.Gt] +
    state[S.GL] +
    state[S.Ki] +
    state[S.RBC] +
    state[S.CumGutI] +
    state[S.CumLivI] +
    state[S.CumFec] +
    state[S.CumHair]
  );
}

function weeklyDoseRateUgPerHour(tHoursMidpoint: number, weeklyDoseUg: number, activeWeeks: number): number {
  if (tHoursMidpoint >= activeWeeks * HOURS_PER_WEEK) return 0;
  const withinWeek = tHoursMidpoint % HOURS_PER_WEEK;
  if (withinWeek >= PAPER_DOSE_DURATION_HOURS) return 0;
  return weeklyDoseUg / PAPER_DOSE_DURATION_HOURS;
}

function simulateTrajectory(input: SimulationInput, activeWeeks: number, totalHours: number): Trajectory {
  const system = buildLinearSystem(input);
  const { params } = system;
  const stepper = buildTrapezoidStepper(system, DT_HOURS);

  const weeklyDoseUg = input.fishPpm * input.mealSizeG; // 1 ppm = 1 ug / g

  const totalSteps = Math.round(totalHours / DT_HOURS);
  let state = new Array(STATE_COUNT).fill(0);
  let cumulativeDoseUg = 0;
  let maxAbsMassBalanceErrorUg = 0;

  const timeHours: number[] = [];
  const timeDays: number[] = [];
  const bloodConc: number[] = [];
  const brainConc: number[] = [];
  const muscleConc: number[] = [];
  const hairConc: number[] = [];

  const sampledTimeHours: number[] = [];
  const sampledTimeDays: number[] = [];
  const sampledBloodConc: number[] = [];
  const sampledBrainConc: number[] = [];
  const sampledMuscleConc: number[] = [];
  const sampledHairConc: number[] = [];
  const sampledMassBalanceErrorUg: number[] = [];

  for (let step = 0; step <= totalSteps; step += 1) {
    const tHours = step * DT_HOURS;
    const tDays = tHours / HOURS_PER_DAY;

    const currentBlood = bloodConcUgPerL(state, params);
    const currentBrain = brainConcUgPerL(state, params);
    const currentMuscle = muscleConcUgPerL(state, params);
    const currentHair = hairConcPpm(state, params);
    const massBalanceErrorUg = cumulativeDoseUg - systemMercuryUg(state);

    maxAbsMassBalanceErrorUg = Math.max(maxAbsMassBalanceErrorUg, Math.abs(massBalanceErrorUg));

    timeHours.push(tHours);
    timeDays.push(tDays);
    bloodConc.push(currentBlood);
    brainConc.push(currentBrain);
    muscleConc.push(currentMuscle);
    hairConc.push(currentHair);

    if (step % SAMPLE_EVERY_STEPS === 0 || step === totalSteps) {
      sampledTimeHours.push(tHours);
      sampledTimeDays.push(tDays);
      sampledBloodConc.push(currentBlood);
      sampledBrainConc.push(currentBrain);
      sampledMuscleConc.push(currentMuscle);
      sampledHairConc.push(currentHair);
      sampledMassBalanceErrorUg.push(massBalanceErrorUg);
    }

    if (step === totalSteps) break;

    const midpointHours = tHours + 0.5 * DT_HOURS;
    const doseRateUgPerHour = weeklyDoseRateUgPerHour(midpointHours, weeklyDoseUg, activeWeeks);
    cumulativeDoseUg += doseRateUgPerHour * DT_HOURS;
    state = stepper.step(state, doseRateUgPerHour);
  }

  const finalMassBalanceErrorUg = cumulativeDoseUg - systemMercuryUg(state);

  return {
    timeHours,
    timeDays,
    bloodConc,
    brainConc,
    muscleConc,
    hairConc,
    sampledTimeHours,
    sampledTimeDays,
    sampledBloodConc,
    sampledBrainConc,
    sampledMuscleConc,
    sampledHairConc,
    sampledMassBalanceErrorUg,
    doseTotalUg: cumulativeDoseUg,
    finalState: state,
    maxAbsMassBalanceErrorUg,
    finalMassBalanceErrorUg,
  };
}

// ---------- summary calculations ----------

function computeHalfLifeDays(singleDose: Trajectory): number {
  const i200 = findClosestIndex(singleDose.timeHours, 200);
  const i300 = findClosestIndex(singleDose.timeHours, 300);
  const c200 = singleDose.bloodConc[i200];
  const c300 = singleDose.bloodConc[i300];
  if (c200 <= 0 || c300 <= 0 || c300 >= c200) return Number.NaN;
  const lambdaPerHour = Math.log(c200 / c300) / (singleDose.timeHours[i300] - singleDose.timeHours[i200]);
  return Math.LN2 / lambdaPerHour / HOURS_PER_DAY;
}

function computeWeeklyAverageBloodConc(repeated: Trajectory): number {
  const weekSteps = Math.round(HOURS_PER_WEEK / DT_HOURS);
  const endIndex = repeated.timeHours.length - 1;
  const startIndex = Math.max(0, endIndex - weekSteps);
  const area = uniformIntegral(repeated.bloodConc, startIndex, endIndex, DT_HOURS);
  return area / HOURS_PER_WEEK;
}

function computeTimeTo95PercentSteadyStateDays(repeated: Trajectory): number {
  const totalWeeks = Math.floor(REPEATED_SIM_HOURS / HOURS_PER_WEEK);
  const weekSteps = Math.round(HOURS_PER_WEEK / DT_HOURS);
  const weeklyAverages: number[] = [];

  for (let week = 0; week < totalWeeks; week += 1) {
    const startIndex = week * weekSteps;
    const endIndex = startIndex + weekSteps;
    const area = uniformIntegral(repeated.bloodConc, startIndex, endIndex, DT_HOURS);
    weeklyAverages.push(area / HOURS_PER_WEEK);
  }

  if (weeklyAverages.length === 0) return Number.NaN;

  const target = 0.95 * weeklyAverages[weeklyAverages.length - 1];
  for (let week = 0; week < weeklyAverages.length; week += 1) {
    if (weeklyAverages[week] >= target) {
      return (week + 1) * 7;
    }
  }

  return totalWeeks * 7;
}

function computeEliminationFractions(singleDose: Trajectory): EliminationFractions {
  const doseUg = singleDose.doseTotalUg;
  const state = singleDose.finalState;

  const gutDemethylationPctDose = (100 * state[S.CumGutI]) / doseUg;
  const hairPctDose = (100 * state[S.CumHair]) / doseUg;
  const liverDemethylationPctDose = (100 * state[S.CumLivI]) / doseUg;
  const fecesPctDose = (100 * state[S.CumFec]) / doseUg;
  const totalEliminatedPctDose =
    gutDemethylationPctDose + hairPctDose + liverDemethylationPctDose + fecesPctDose;

  const pctOfEliminated = (valuePctDose: number): number =>
    totalEliminatedPctDose > 0 ? (100 * valuePctDose) / totalEliminatedPctDose : 0;

  return {
    gutDemethylationPctDose,
    hairPctDose,
    liverDemethylationPctDose,
    fecesPctDose,
    totalEliminatedPctDose,
    gutDemethylationPctEliminated: pctOfEliminated(gutDemethylationPctDose),
    hairPctEliminated: pctOfEliminated(hairPctDose),
    liverDemethylationPctEliminated: pctOfEliminated(liverDemethylationPctDose),
    fecesPctEliminated: pctOfEliminated(fecesPctDose),
  };
}

// ---------- public interface ----------

export function runSimulation(input: SimulationInput): SimulationResult {
  const repeated = simulateTrajectory(input, REPEATED_DOSE_WEEKS, REPEATED_SIM_HOURS);
  const singleDose = simulateTrajectory(input, 1, SINGLE_DOSE_SIM_HOURS);

  return {
    timeHours: repeated.sampledTimeHours,
    timeDays: repeated.sampledTimeDays,
    bloodConc: repeated.sampledBloodConc,
    brainConc: repeated.sampledBrainConc,
    muscleConc: repeated.sampledMuscleConc,
    hairConc: repeated.sampledHairConc,
    halfLifeDays: Number(computeHalfLifeDays(singleDose).toFixed(1)),
    eqTimeDays: Number(computeTimeTo95PercentSteadyStateDays(repeated).toFixed(0)),
    peakBloodConc: Number(Math.max(...singleDose.bloodConc).toFixed(3)),
    steadyStateBloodConc: Number(computeWeeklyAverageBloodConc(repeated).toFixed(3)),
    elimination50d: computeEliminationFractions(singleDose),
    maxAbsMassBalanceErrorUg: repeated.maxAbsMassBalanceErrorUg,
    finalMassBalanceErrorUg: repeated.finalMassBalanceErrorUg,
  };
}
