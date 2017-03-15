#=
Solves Khan and Thomas with aggregate shocks in continuous time
Method is as described in Ahn, Kaplan, Moll, and Winberry (2015)
Written by:    Victoria Gregory
Date:          4/17/2016
=#

using ForwardDiff
using PyPlot

#---------------------------------------------
# Define model type containing all parameters
#---------------------------------------------

type KTModel
  # Economic parameters:
  σ::Float64    # risk aversion coefficient
  ρ::Float64    # discount rate
  ϕ::Float64    # Frisch elasticity
  Θ::Float64    # capital coefficient
  ν::Float64    # labor coefficient
  δ::Float64    # depreciation rate
  χ0::Float64   # capital adj. costs
  χ1::Float64
  σTFP::Float64 # std. dev of TFP shocks
  ρTFP::Float64 # autocorr. of TFP shocks

  # Approximation parameters:
  nE::Int64              # grid sizes
  nK::Int64

  ϵ::Vector{Float64}     # productivity shocks
  ee::Matrix{Float64}
  eee::Vector{Float64}
  λ1::Float64
  λ2::Float64
  llaE::Vector{Float64}
  eAvg::Float64

  kmin::Float64         # capital grid
  kmax::Float64
  k::Vector{Float64}
  kk::Matrix{Float64}
  kkk::Vector{Float64}
  dk::Float64

  mLamF::SparseMatrixCSC{Float64,Int64}  # useful for computations later

  wmin::Float64           # for steady state computations
  wmax::Float64
  w0::Float64
  crit::Float64
  Δ::Float64
  maxit::Int64

  nVars::Int64           # Number of endogenous variables
  nEErrors::Int64
end

#---------------------------------------------------
# Define type for steady state solution of the model
#---------------------------------------------------
type KTss
  w::Float64
  output::Float64
  sdf::Float64
  v::Matrix{Float64}
  g::Matrix{Float64}
  If::BitArray
  Ib::BitArray
  I0::Matrix{Int64}
  investment::Matrix{Float64}
  vars::Vector{Float64}
end

#-------------------------------------------------
# Define type for all aggregate variables along simulation
#-------------------------------------------------

type Aggregates

  Investment::Float64
  Capital::Float64
  Consumption::Float64
  Output::Float64
  Hours::Float64
  Wage::Float64
  SDF::Float64
  Dist1::Vector{Float64}
  Dist2::Vector{Float64}
  InvPol1::Vector{Float64}
  InvPol2::Vector{Float64}

end

#---------------------------------------------------
# Define type for the state vector
#---------------------------------------------------
type StateVec
  v
  g
  g_end
  sdf
  logAggregateTFP
end

#---------------------------------------------------
# Function that sets up the Khan/Thomas model
#---------------------------------------------------

function KTModel(;σ::Float64=2.0,
  ρ::Float64=0.01,
  ϕ::Float64=0.5,
  Θ::Float64=0.21,
  ν::Float64=0.64,
  δ::Float64=0.025,
  χ0::Float64=0.001,
  χ1::Float64=2.0,
  σTFP::Float64=0.007,
  ρTFP::Float64=0.95,
  nE::Int64=2,
  nK::Int64=100,
  ϵ::Vector{Float64}=[0.9, 1.1],
  λ1::Float64=0.25,
  λ2::Float64=0.25,
  crit::Float64=1e-6,
  Δ::Float64=1.0e4,
  maxit::Int64=100)

  # set up space for idiosyncratic productivity shocks
  ee = ones(nK,1)*ϵ'
  eee = vec(reshape(ee,nE*nK,1))
  llaE = vec([λ1, λ2])
  eAvg = (llaE[1]*ϵ[2] + llaE[2]*ϵ[1])/(llaE[1] + llaE[2])

  # solve for representative agent steady state
  kRepSS = ((eAvg * Θ * ((1/3)) ^ ν) / (ρ + δ)) ^ (1 / (1 - Θ));
  wRepSS = eAvg * (kRepSS ^ Θ) * ν * ((1/3)) ^ (ν - 1);
  yRepSS = eAvg * (kRepSS ^ Θ) * (((1/3)) ^ ν)

  # set-up the capital grid
  kmin = 0.8*kRepSS
  kmax = 1.2*kRepSS
  k = linspace(kmin, kmax, nK)
  k = collect(k)
  kk = k*ones(1,nE)
  kkk = vec(reshape(kk,nE*nK,1))
  dk = (kmax - kmin)/(nK-1)

  # for computation
  mLamF = [-speye(nK) * llaE[1] speye(nK) * llaE[1]; speye(nK) * llaE[2] -speye(nK) * llaE[2]];

  # steady state computations
  wmin = 0.8*wRepSS
  wmax = 1.2*wRepSS
  w0 = wRepSS

  # number of endogenous variables
  nVars = 2*nE*nK-1+2
  nEErrors = nE*nK

  # put everything into a KTModel type
  KTModel(σ, ρ, ϕ, Θ, ν, δ, χ0, χ1, σTFP, ρTFP, nE, nK, ϵ,
  ee, eee, λ1, λ2, llaE, eAvg, kmin, kmax, k, kk, kkk, dk,
  mLamF, wmin, wmax, w0, crit, Δ, maxit, nVars, nEErrors)
end

#--------------------------------------------------
# Function that computes aggregate variables
#--------------------------------------------------

function Aggregates(eg::KTModel, eg_ss::KTss, state::StateVec, w::Float64, χ::Float64, vLaborDemand::Matrix{Float64}, investment::Matrix{Float64}, k, ϵ)

  Investment = sum(investment .* [state.g; state.g_end] * eg.dk)
  Capital = sum(k .* [state.g; state.g_end] * eg.dk)
  vConsumptionGrid = (exp(state.logAggregateTFP) * ϵ .* (k .^ eg.Θ) .* (reshape(vLaborDemand,eg.nK*eg.nE,1) .^ eg.ν)) - investment - eg.χ0 * (abs(investment) .> 1e-8) - (eg.χ1 / 2) * ((investment ./ k) .^ 2) .* k
  Consumption = sum(vConsumptionGrid .* [state.g; state.g_end] * eg.dk)
  vOutputGrid = (exp(state.logAggregateTFP) * ϵ .* (k .^ eg.Θ) .* (reshape(vLaborDemand,eg.nK*eg.nE,1) .^ eg.ν))
  Output = sum(vOutputGrid[:] .* [state.g; state.g_end] * eg.dk)
  Hours = (w / χ) ^ (1 / eg.ϕ)
  Wage = w
  SDF = state.sdf
  Dist1 = state.g[1:eg.nK]
  Dist2 = [state.g[eg.nK+1:2*eg.nK-1]; state.g_end]
  InvPol1 = investment[1:eg.nK]
  InvPol2 = investment[eg.nK+1:2*eg.nK]

  Aggregates(Investment, Capital, Consumption, Output, Hours, Wage, SDF, vec(Dist1), vec(Dist2), vec(InvPol1), vec(InvPol2))

end

#---------------------------------------------------
# Function that extracts the state from a Vector
#---------------------------------------------------
function extract(vars, eg::KTModel, eg_ss::KTss)

  V = vars[1:eg.nE*eg.nK, 1] + eg_ss.vars[1:eg.nE*eg.nK, 1]
  V = reshape(V, eg.nK, eg.nE)
  g = vars[eg.nE*eg.nK+1:2*eg.nE*eg.nK-1] + eg_ss.vars[eg.nE*eg.nK+1:2*eg.nE*eg.nK-1]
  g_end = 1 / eg.dk - sum(g)
  sdf = vars[2*eg.nE*eg.nK, 1] + eg_ss.vars[2*eg.nE*eg.nK, 1]
  logAggregateTFP = vars[2*eg.nE*eg.nK+1, 1]

  return StateVec(V, g, g_end, sdf, logAggregateTFP)

end

#-------------------------------------------------
# Define some useful functions
#-------------------------------------------------

# individual labor function (from FOC)
function labor_demand(eg::KTModel, w::Float64, k::Matrix{Float64}, ϵ::Matrix{Float64})
  n_i = (w ./ (ϵ .* k .^eg.Θ .* eg.ν)) .^ (1 / (eg.ν - 1))
  return n_i
end

# envelope condition: π' = v'
function envelope(eg::KTModel, k::Vector{Float64}, i::Vector{Float64})
  env = 1 + (eg.χ1 .* i)./k
  return env
end

# drift of capital
function kdot(eg::KTModel, k::Matrix{Float64}, i::Matrix{Float64})
  kd = i - eg.δ .* k
  return kd
end

# optimal investment, given value function and capital
function ipol(eg::KTModel, dV::Matrix{Float64}, k::Matrix{Float64})
  i = (k./eg.χ1) .* (dV - 1)
  opt_i = i .* (dV.*i - (i + eg.χ0 + (eg.χ1/2).*((i./k).^2) .* k) .>= 0)
  netInv = kdot(eg, eg.kk, opt_i)
  return opt_i, netInv
end

# optimal investment for case with aggregate shocks
function ipol_agg(eg::KTModel, state::StateVec, dV, k)
  inv = (k / eg.χ1) .* ((reshape(dV,eg.nE*eg.nK,1) * (state.sdf ^ (-1))) - 1)
  inv = inv .* (reshape(dV,eg.nE*eg.nK,1) .* inv - state.sdf * (inv + eg.χ0 + (eg.χ1 / 2) * ((inv ./ k) .^ 2) .* k) .>= 0)
  netInv = reshape(inv - eg.δ * k, eg.nK, eg.nE)
  return inv, netInv
end

function wage(eg::KTModel, eg_ss::KTss, state::StateVec, k, ϵ)
  vIntegrandGrid = (ϵ .* (k .^ eg.Θ)) .^ (1 / (1 - eg.ν))
  integral = sum(vIntegrandGrid[1:eg.nE*eg.nK-1] .* state.g * eg.dk) + state.g_end * eg.dk * vIntegrandGrid[eg.nE*eg.nK]
  χ = eg_ss.w / ((1/3) ^ eg.ϕ)
  w = ((χ ^ (1 - eg.ν)) * ((exp(state.logAggregateTFP) * eg.ν) ^ eg.ϕ) * (integral ^ (eg.ϕ * (1 - eg.ν)))) ^ (1 / (eg.ϕ + 1 - eg.ν))
  return w, χ
end

# aggregate labor
function agg_labor(eg::KTModel,state::StateVec, w, k, ϵ)
  vLaborDemand = ((eg.ν * exp(state.logAggregateTFP) * ϵ .* (k .^ eg.Θ)) * (w ^ (-1))) .^ (1 / (1 - eg.ν))
  vLaborDemand = reshape(vLaborDemand, eg.nK, eg.nE)
  return vLaborDemand
end

#-------------------------------------------------
# Function to compute the steady state
# (no aggregate shocks case)
#-------------------------------------------------

function steady_state(eg::KTModel)

  # initialize first wage guess
  w = eg.w0

  dVf = zeros(eg.nK, eg.nE)
  dVb = zeros(eg.nK, eg.nE)

  # stuff we'll need to access outside the inner for loop
  A = spzeros(eg.nK*2, eg.nK*2)
  V = zeros(eg.nK, eg.nE)
  v = zeros(eg.nK, eg.nE)
  g = zeros(eg.nK, eg.nE)
  investment = zeros(eg.nK, eg.nE)
  If = (investment .> 0)
  Ib = (investment .> 0)
  I0 = (investment .> 0)

  for i = 1:eg.maxit

      # labor demand
      vLaborDemand = labor_demand(eg, w, eg.kk, eg.ee)

      # guess for value function
      v0 = (eg.ee .* (eg.kk .^ eg.Θ) .* (vLaborDemand .^ eg.ν) - w * vLaborDemand - eg.δ * eg.kk - eg.χ0 - (eg.χ1/2) * (eg.δ ^ 2))/eg.ρ

      # if not the first wage guess, use the value function from last time
      if i > 1
        v0 = V
      end
      v = v0

      # solve for the value function given w

      for n = 1:eg.maxit

        V = v;

        # compute forward difference
        dVf[1:eg.nK-1, :] = (V[2:eg.nK, :] - V[1:eg.nK-1, :]) / eg.dk
        dVf[eg.nK, :] = envelope(eg, eg.kk[end,:], eg.kk[end,:].*eg.δ)

        # compute backward difference
        dVb[2:eg.nK, :] = (V[2:eg.nK, :] - V[1:eg.nK-1, :]) / eg.dk
        dVb[1, :] = envelope(eg, eg.kk[1,:], eg.kk[1,:].*eg.δ)

        # compute investment and drift with forward difference
        invF, netInvF = ipol(eg, dVf, eg.kk)

        # compute investment and drift with backward difference
        invB, netInvB = ipol(eg, dVb, eg.kk)

        # derivative of value function with no drift
        dV0 = envelope(eg, ones(eg.nK, eg.nE), ones(eg.nK, eg.nE).*eg.δ)

        # upwind method
        If = (netInvF .> 1e-20)
        Ib = (netInvB .< 1e-20)
        I0 = (1 - If - Ib)
        dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0

        # compute investment from upwind method
        investment, netInv = ipol(eg, dV_Upwind, eg.kk)
        u = eg.ee .* (eg.kk .^ eg.Θ) .* (vLaborDemand .^ eg.ν) - w * vLaborDemand  - investment - eg.χ0 * (abs(investment) .> 1e-8) - (eg.χ1 / 2) * ((investment ./ eg.kk) .^ 2) .* eg.kk

        # matrices for implicit scheme
        X = -min(netInvB, 0) ./ eg.dk
        Y = -max(netInvF, 0) ./ eg.dk + min(netInvB, 0) ./ eg.dk
        Z = max(netInvF, 0) ./ eg.dk

        A1 = spdiagm(Y[:,1], 0, eg.nK, eg.nK) + spdiagm(X[2:eg.nK,1], -1, eg.nK, eg.nK) + spdiagm(Z[1:eg.nK-1,1], 1, eg.nK, eg.nK)
        A2 = spdiagm(Y[:,2], 0, eg.nK, eg.nK) + spdiagm(X[2:eg.nK,2], -1, eg.nK, eg.nK) + spdiagm(Z[1:eg.nK-1,2], 1, eg.nK, eg.nK)
        A = [A1 spzeros(eg.nK, eg.nK); spzeros(eg.nK,eg.nK) A2] + eg.mLamF

        B = (1/eg.Δ + eg.ρ)*speye(2*eg.nK) - A

        u_stacked = reshape(u, eg.nE*eg.nK, 1)
        V_stacked = reshape(V, eg.nE*eg.nK, 1)
        b = u_stacked + V_stacked/eg.Δ

        # solve the system
        V_stacked = B\b

        V = reshape(V_stacked, eg.nK, eg.nE)

        # check for convergence
        Vchange = V - v
        v = V

        dist = maximum(maximum(abs(Vchange)))
        if dist < eg.crit
          @printf("Value function converged, iteration %d.\n", n)
          break
        end
      end

      ###
      # Solve KF equation for stationary distribution
      ###

      AT = A'
      b = zeros(2*eg.nK, 1)

      # pdf Normalization
      i_fix = 1
      b[i_fix] = 0.1
      row = [zeros(1, i_fix-1) 1 zeros(1, 2*eg.nK-i_fix)]
      AT[i_fix, :] = row

      # solve linear system
      gg = AT\b
      g_sum = gg'*ones(eg.nE*eg.nK, 1)*eg.dk
      gg = gg./g_sum
      g = reshape(gg, eg.nK, eg.nE)

      # compute excess labor supply
      S = (1/3) - sum(reshape(vLaborDemand, eg.nE*eg.nK, 1) .* gg * eg.dk)

      # update wage
      if S > eg.crit
        eg.wmax = w
        w = 0.5*(w + eg.wmin)

      elseif S < -eg.crit
        eg.wmin = w
        w = 0.5*(w + eg.wmax)

      elseif abs(S) < eg.crit
        @printf("Steady state wage found: %.3f.\n", w)

        # compute some aggregates
        χ = w / ((1/3) ^ eg.ϕ)
        vOutputGrid = eg.ee .*(eg.kk .^ eg.Θ) .* (vLaborDemand .^ eg.ν)
        vConsumptionGrid = vOutputGrid - investment - eg.χ0 * (abs(investment) .> 1e-8) - (eg.χ1/2) * ((investment ./ eg.kk) .^2) .* eg.kk
        output = sum(vOutputGrid[:] .* g[:] * eg.dk)
        consumption = sum(vConsumptionGrid[:] .* g[:] * eg.dk)
        sdf = (consumption - χ * ((1/3) ^ (1 + eg.ϕ)) / (1 + eg.ϕ)) .^ (-eg.σ)

        # combine steady state variables for input into equilibrium conditions function
        varsSS = zeros(eg.nVars, 1)
        varsSS[1:eg.nE*eg.nK, 1] = sdf * reshape(v, eg.nE*eg.nK, 1)
        ggSS = reshape(g, eg.nE*eg.nK, 1)
        varsSS[eg.nE*eg.nK+1:2*eg.nK*eg.nE-1, 1] = ggSS[1:eg.nE*eg.nK-1, 1]
        varsSS[2*eg.nE*eg.nK, 1] = sdf
        varsSS[2*eg.nE*eg.nK+1, 1] = 0

        return KTss(w, output, sdf, v, g, If, Ib, I0, investment, vec(varsSS))
        break
      end

  end

end


# create KTModel object
eg = KTModel()
#@time vSS, gSS, wSS, outputSS, sdfSS, IfSS, IbSS, I0SS = steady_state(eg)
@time eg_ss = steady_state(eg)

vars = vec(zeros(2*eg.nVars + eg.nEErrors +1, 1))

#------------------------------------------------------
# Function to compute equilibrium conditions in gensys form
# -----------------------------------------------------

function equilibrium_conditions(vars::Vector)

  # unpack the input

  # everything in y
  state = extract(vars[1:eg.nVars], eg, eg_ss)

  # everything in ydot
  vDot = vars[eg.nVars+1:eg.nVars+eg.nE*eg.nK, 1]
  gDot = vars[eg.nVars+eg.nE*eg.nK+1:eg.nVars+2*eg.nE*eg.nK-1, 1]
  sdfDot = vars[eg.nVars+2*eg.nE*eg.nK, 1]
  logAggregateTFPDot = vars[eg.nVars+2*eg.nE*eg.nK+1, 1]

  # everything in η
  VEErrors = vars[2*eg.nVars+1:2*eg.nVars+eg.nE*eg.nK, 1]

  # everything in z
  aggregateTFPShock = vars[2*eg.nVars+eg.nEErrors+1, 1]

  dVf = zeros(state.v)
  dVb = zeros(state.v)

  # compute wage
  w, χ = wage(eg, eg_ss, state, eg.kkk, eg.eee)

  # firm HJB
  vLaborDemand = agg_labor(eg, state, w, eg.kkk, eg.eee)

  # forward difference
  dVf[1:eg.nK-1,:] = (state.v[2:eg.nK,:] - state.v[1:eg.nK-1,:]) / eg.dk
  dVf[eg.nK,:] = (state.sdf * (1 + eg.δ * eg.χ1)).*ones(1,eg.nE)

  # backward difference
  dVb[2:eg.nK,:] = (state.v[2:eg.nK,:] - state.v[1:eg.nK-1,:]) / eg.dk
  dVb[1,:] = (state.sdf * (1 + eg.δ * eg.χ1)).*ones(1,eg.nE)

  # investment policy with forward difference
  invF, netInvF = ipol_agg(eg, state, dVf, eg.kkk)

  # investment policy with backward difference
  invB, netInvB = ipol_agg(eg, state, dVb, eg.kkk)

  # derivative of value function with no drift
  dV0 = state.sdf * (1 + eg.δ * eg.χ1) * ones(eg.nK,eg.nE)

  # upwind method
  dV_Upwind = dVf .* eg_ss.If + dVb .* eg_ss.Ib + dV0 .* eg_ss.I0

  # investment using upwind method
  investment, netInv = ipol_agg(eg, state, dV_Upwind, eg.kkk)

  # profits
  u = state.sdf * (exp(state.logAggregateTFP) * eg.eee .* (eg.kkk .^ eg.Θ) .* (reshape(vLaborDemand,eg.nK*eg.nE,1) .^ eg.ν) - w * reshape(vLaborDemand,eg.nK*eg.nE,1) - investment - eg.χ0 * (abs(investment) .> 1e-8) - (eg.χ1 / 2) * ((investment ./ eg.kkk) .^ 2) .* eg.kkk)

  # construct A matrix
  X = -netInvB.*eg_ss.Ib/eg.dk
  Y = -netInvF.*eg_ss.If/eg.dk + netInvB.*eg_ss.Ib/eg.dk
  Z = netInvF.*eg_ss.If/eg.dk
  X[1,:]=0
  lowdiag = reshape(X,eg.nE*eg.nK,1)
  Z[eg.nK,:]=0
  updiag = reshape(Z,eg.nE*eg.nK,1)
  A = spdiagm(vec(reshape(Y,eg.nE*eg.nK,1)),0,eg.nE*eg.nK,eg.nE*eg.nK) + spdiagm(lowdiag[2:eg.nE*eg.nK],-1,eg.nE*eg.nK,eg.nE*eg.nK) + spdiagm(updiag[1:eg.nE*eg.nK-1],1,eg.nE*eg.nK,eg.nE*eg.nK) + eg.mLamF

  # equilibrium conditions

  # firm HJB
  hjbResidual = reshape(u,eg.nE*eg.nK,1) + A * reshape(state.v,eg.nE*eg.nK,1) + vDot - VEErrors - eg.ρ * reshape(state.v,eg.nE*eg.nK,1)

  # firm KFE
  gDotIntermediate = A' * [state.g; state.g_end]
  gResidual = gDot - gDotIntermediate[1:eg.nE*eg.nK-1,1]

  # sdf
  vConsumptionGrid = (exp(state.logAggregateTFP) * eg.eee .* (eg.kkk .^ eg.Θ) .* (reshape(vLaborDemand,eg.nK*eg.nE,1) .^ eg.ν)) - investment - eg.χ0 * (abs(investment) .> 1e-8) - (eg.χ1 / 2) * ((investment ./ eg.kkk) .^ 2) .* eg.kkk
  consumption = sum(vConsumptionGrid .* [state.g; state.g_end] * eg.dk)
  labor = (w / χ) ^ (1 / eg.ϕ)
  sdfResidual = (consumption - χ * ((labor ^ (1 + eg.ϕ)) / (1 + eg.ϕ))) ^ (-eg.σ) - state.sdf

  # aggregate shocks
  tfpResidual = logAggregateTFPDot + (1 - eg.ρTFP) * state.logAggregateTFP - eg.σTFP * aggregateTFPShock

  vResidual = [hjbResidual; gResidual; sdfResidual; tfpResidual]

  return vec(vResidual)

end

# compute Jacobian of equilibrium_conditions at steady state
@time derivs = ForwardDiff.jacobian(equilibrium_conditions, vars)

# unpackage the partial derivatives
mVarsDerivs = derivs[:,1:eg.nVars];
mVarsDotDerivs = derivs[:,eg.nVars+1:2*eg.nVars]
mEErrorsDerivs = derivs[:,2*eg.nVars+1:2*eg.nVars+eg.nEErrors]
mShocksDerivs = derivs[:,2*eg.nVars+eg.nEErrors+1]

# re-arrange into gensys form
Γ0 = mVarsDotDerivs
Γ1 = -mVarsDerivs
c = spzeros(eg.nVars, 1)
Ψ = -mShocksDerivs
Π= -mEErrorsDerivs

# solve system using gensys
include("gensys.jl")
@time G1, impact = gensys(Γ0, Γ1, c, Ψ, Π)

#------------------------------------------------------------
# Simulate the model
#------------------------------------------------------------

function simulate(eg::KTModel, eg_ss::KTss, vAggregateShock, G1, impact, N, dt)

  # endogenous variables
  vVarsSeries = zeros(eg.nVars, N+1)

  # time series for aggregates
  agg_series = Array(Aggregates, N)

  # time series for endogenous state
  state_series = Array(StateVec, N+1)
  state_series[1] = extract(zeros(eg.nVars, 1), eg, eg_ss)

  # simulate
  for n = 1 : N

    # endogenous variables
    vVarsSeries[:, n+1] = (dt * G1 + speye(eg.nVars)) * vVarsSeries[:,n] + (dt ^ (1 / 2)) * impact * vAggregateShock[n,1]'

    # extract relevant objects
    vars = vVarsSeries[:, n+1]
    state = extract(vars, eg, eg_ss)
    state_series[n+1] = state
    dVf = zeros(state.v)
    dVb = zeros(state.v)

    # Wage
    w, χ = wage(eg, eg_ss, state, eg.kkk, eg.eee)

    # Labor demand
    vLaborDemand = agg_labor(eg, state, w, eg.kkk, eg.eee)

    # Compute forward difference
	  dVf[1:eg.nK-1,:] = (state.v[2:eg.nK,:] - state.v[1:eg.nK-1,:]) / eg.dk
	  dVf[eg.nK,:] = (state.sdf * (1 + eg.δ * eg.χ1)).*ones(1,eg.nE)

	  # Compute backward difference
	  dVb[2:eg.nK,:] = (state.v[2:eg.nK,:] - state.v[1:eg.nK-1,:]) / eg.dk
	  dVb[1,:] = (state.sdf * (1 + eg.δ * eg.χ1)).*ones(1,eg.nE)

    # Compute investment policy with forward difference
    invF, netInvF = ipol_agg(eg, state, dVf, eg.kkk)

    # Compute investment policy with backward difference
    invB, netInvB = ipol_agg(eg, state, dVb, eg.kkk)

    # Compute derivative of value function with no drift
	  dV0 = state.sdf * (1 + eg.δ * eg.χ1) * ones(eg.nK,eg.nE)

    # Compute upwind differences
	  dV_Upwind = dVf .* eg_ss.If + dVb .* eg_ss.Ib + dV0 .* eg_ss.I0
	  investment, netInv = ipol_agg(eg, state, dV_Upwind, eg.kkk)

    # compute aggregates
    aggs = Aggregates(eg, eg_ss, state, w, χ, vLaborDemand, investment, eg.kkk, eg.eee)
    agg_series[n] = aggs

  end

  return agg_series, state_series
end

# time discretization
T = 200
dt = 0.1
N = T/dt
N = round(Int, N)

vAggregateShock = zeros(N, 1)
vAggregateShock[1:convert(Int64, 1/dt), 1] = 0.1

@time agg_series, state_series = simulate(eg, eg_ss, vAggregateShock, G1, impact, N, dt)

#---------------------------------------------
# Plots
#---------------------------------------------

# steady state objects

fig, axes = subplots(1,2)
ax = axes[1]
ax[:plot](eg.k, eg_ss.investment[:,1], "b--", lw = 2, label = L"\varepsilon = \varepsilon_L")
ax[:plot](eg.k, eg_ss.investment[:,2], "r-", lw = 2, label = L"\varepsilon = \varepsilon_H")
ax[:legend](ncol = 1, fontsize = 12)
ax[:set_xlabel]("Capital stock")
ax[:set_title]("Investment Policies")

ax = axes[2]
ax[:plot](eg.k, eg_ss.g[:,1], "b--", lw = 2, label = L"\varepsilon = \varepsilon_L")
ax[:plot](eg.k, eg_ss.g[:,2], "r-", lw = 2, label = L"\varepsilon = \varepsilon_H")
ax[:legend](ncol = 1, fontsize = 12)
ax[:set_xlabel]("Capital stock")
ax[:set_ylim]([0, 1.6])
ax[:set_title]("Steady State Distributions")

# compute some impulse responses

inv_irf = zeros(N,1)
cap_irf = zeros(N,1)
cons_irf = zeros(N,1)
out_irf = zeros(N,1)
hrs_irf = zeros(N,1)
wage_irf = zeros(N,1)

for n = 1:N
    inv_irf[n] = 100*(agg_series[n].Investment - agg_series[N].Investment)/agg_series[N].Investment
    cap_irf[n] = 100*(agg_series[n].Capital - agg_series[N].Capital)/agg_series[N].Capital
    cons_irf[n] = 100*(agg_series[n].Consumption - agg_series[N].Consumption)/agg_series[N].Consumption
    out_irf[n] = 100*(agg_series[n].Output - agg_series[N].Output)/agg_series[N].Output
    hrs_irf[n] = 100*(agg_series[n].Hours - agg_series[N].Hours)/agg_series[N].Hours
    wage_irf[n] = 100*(agg_series[n].Wage - agg_series[N].Wage)/agg_series[N].Wage
end

t = linspace(1, T, N)

fig, axes = subplots(3,2, figsize = (10,10))
ax = axes[1]
ax[:plot](t, inv_irf, "b-", lw = 2)
ax[:set_title]("Investment")
ax[:set_ylabel]("% Deviation from Steady State", fontsize = 10)
ax[:set_ylim]([0, 1.6])

ax = axes[2]
ax[:plot](t, cap_irf, "b-", lw = 2)
ax[:set_title]("Capital")
ax[:set_ylabel]("% Deviation from Steady State", fontsize = 10)
ax[:set_ylim]([0, 0.25])

ax = axes[3]
ax[:plot](t, cons_irf, "b-", lw = 2)
ax[:set_title]("Consumption")
ax[:set_xlabel]("Quarters")
ax[:set_ylabel]("% Deviation from Steady State", fontsize = 10)

ax = axes[4]
ax[:plot](t, out_irf, "b-", lw = 2)
ax[:set_title]("Output")

ax = axes[5]
ax[:plot](t, hrs_irf, "b-", lw = 2)
ax[:set_title]("Hours")

ax = axes[6]
ax[:plot](t, wage_irf, "b-", lw = 2)
ax[:set_xlabel]("Quarters")
ax[:set_title]("Wage")

savefig("irfs.pdf")
