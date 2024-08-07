;Function uses the magnetopause model from Chao et al. Advances in Space Research 2001
;to output return x and y coordinates on the equitorial plane in GSM
; Inputs: 	IMF bz in nT
;			Dynamic pressure in nPa


FUNCTION chao_plane, bz,pdyn
result = fltarr(2,356)
Xm=fltarr(356)
Ym=fltarr(356)

a1 = 11.646
a2 = 0.216
a3 = 0.122
a4 = 6.215
a5 = 0.578
a6 = -0.009
a7 = 0.012
alpha = (a5+a6*bz)*(1+a7*pdyn)

if bz ge 0 then begin
  ro = a1*pdyn^(-1./a4)
endif
if ((bz ge -8) and (bz lt 0)) then begin
  ro = (a1+a2*bz)*pdyn^(-1./a4)
endif
if ((bz lt -8)) then begin
  ro = (a1+a2*bz)*pdyn^(-1./a4)
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