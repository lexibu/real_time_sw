;Function uses the magnetopause model from Yang et al. JGR 2003
; This model accounts for a saturation in a Bz impact on magnetosphere
;to output return x and y coordinates on the equitorial plane in GSM
; Inputs: 	IMF bz in nT
;			Dynamic pressure in nPa


FUNCTION yang_plane, bz,pdyn
result = fltarr(2,356)
Xm=fltarr(356)
Ym=fltarr(356)

bzp = bz
lim = -8.1-12.0*alog(pdyn+1)
if bzp lt lim then bzp = lim

a1 = 11.646
a2 = 0.216
a3 = 0.122
a4 = 6.215
a5 = 0.578
a6 = -0.009
a7 = 0.012
a7 = a7*exp(-1*pdyn/30.)
alpha = (a5+a6*bzp)*(1+a7*pdyn)

if bzp ge 0 then begin
  ro = a1*pdyn^(-1./a4)
endif
if ((bzp ge -8) and (bzp lt 0)) then begin
  ro = (a1+a2*bzp)*pdyn^(-1./a4)
endif
if ((bzp lt -8)) then begin
  ro = (a1+a2*bzp)*pdyn^(-1./a4)
endif
;ro=(10.22+1.29*tanh(0.184*(bz+8.14)))*(pdyn)^(-1/6.6)
;alpha=(0.58-0.007*bz)*(1+0.024*ALOG(pdyn))


; Loop through different angles to get x and y values of magnetopause
FOR i=0, 179 DO BEGIN
	theta=2*!PI*i/(360)
	r=ro*(2/(1+COS(theta)))^alpha

    Xm[i]=r*COS(theta)
    Ym[i]=r*SIN(theta)
ENDFOR

FOR i=180, 355 DO BEGIN
    theta=2*!PI*(i+5)/(360)
    r=ro*(2/(1+COS(theta)))^alpha

    Xm[i]=r*COS(theta)
    Ym[i]=r*SIN(theta)
ENDFOR

result[0,*] = Xm
result[1,*] = Ym

return, result
END