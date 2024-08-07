;Function uses the magnetopause model from Shue et al. JGR 1998
;to output radial distance at a given GSM x, y, z in RE
;			Dynamic pressure in nPa


FUNCTION r_shue, bz,pdyn, x,y,z

  theta = atan(sqrt(z^2+y^2),x)
  ro=(10.22+1.29*tanh(0.184*(bz+8.14)))*(pdyn)^(-1/6.6)
  alpha=(0.58-0.007*bz)*(1+0.024*ALOG(pdyn))
  r=ro*(2/(1+COS(theta)))^alpha
  
  return, r
END