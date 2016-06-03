#---------------------------------------------------
# Implements gensys for a continuous time process
# Note: doesn't check for existence/uniqueness
#---------------------------------------------------

function gensys(Γ0, Γ1, c, Ψ, Π)

  # put into reduced form
  temp = (maximum(abs([Γ0 Ψ]), 2) .== 0)
  redundant = find(temp)
  base = nullspace(Γ1[redundant, :])

  Γ0 = base'*Γ0*base
  Γ1 = base'*Γ1*base
  Γ1 = Γ0\Γ1
  Ψ = Γ0\base'*Ψ
  Π = Γ0\base'*Π
  c = Γ0\base'*c

  # Schur factorization, re-order eigenvalues
  n = size(Γ1, 1)
  Γ1 = schurfact!(Γ1)
  select = real(Γ1[:values]) .< 0
  ordschur!(Γ1, select)
  nunstab = sum(select)
  U = Γ1[:vectors]
  T = Γ1[:Schur]

  # compute G1
  G1 = U*T*spdiagm(vec([ones(n-nunstab, 1); zeros(nunstab, 1)]), 0, n, n)*U'
  G1 = real(G1)
  G1 = base*G1*base'

  # compute impact
  u2=U[:,n-nunstab+1:n]'
  u1=U[:,1:n-nunstab]'

  etawt = u2*Π
  etawt_fac = svdfact!(etawt)
  ueta, deta, veta  = etawt_fac[:U], etawt_fac[:S], etawt_fac[:V]
  md = minimum(size(deta))
  realsmall = sqrt(eps())*10
  bigev = find((deta[1:md]) .> realsmall)
  ueta = ueta[:, bigev]
  veta = veta[:, bigev]
  deta = deta[bigev]

  zwt = u2*Ψ
  impact = real(-Π * veta * (diagm(deta)\ueta')*u2*Ψ + Ψ)
  impact = base*impact

  return G1, impact

end
