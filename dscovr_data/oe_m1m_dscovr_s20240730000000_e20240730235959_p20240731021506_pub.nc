CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240730000000_e20240730235959_p20240731021506_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-31T02:15:06.708Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-30T00:00:00.000Z   time_coverage_end         2024-07-30T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy��  J          AU��G�@W
=A9G�Bs{C����G�@�ffA3\)Bfz�C@                                     By&  @          AZ�\���@���A2{B]p�C����@�{A*ffBO�C5�                                    By�  J          A[����@I��A.=qBT33CaH���@�A(z�BJ��C�                                    By(r  @          AU��У�@XQ�A/\)B`�RC�
�У�@�p�A)G�BUffC޸                                    By7  
�          AW33���@�A�B<=qC%�{���@Mp�A��B5�C �                                    ByE�            AVff���@��A+\)BVQ�C�3���@�=qA!��BEffC +�                                    ByTd  �          AV=q���\@�  A-��BX��B�L����\@���A"�RBF
=B���                                    Byc
  �          A[33��Q�@ȣ�A7
=Bc�RB�=q��Q�@�33A,Q�BP�B�B�                                    Byq�  �          A^ff���R@��HAB�HBv��C�R���R@�  A:�HBf{CG�                                    By�V  T          A[
=��
=@���A7�
Bd�\C� ��
=@��
A.�RBS�HC^�                                    By��  "          A[\)����@��A7�
Bd�Ch�����@�p�A0(�BV�\C
�                                    By��  
Z          AW�
��{@�z�A0  B[�C(���{@�{A'�
BM{C	c�                                    By�H  T          AY���@���A�B1�C���@�AffB$z�C��                                    By��  T          A^ff��@�ffA�
B8z�C  ��@��A33B,
=CT{                                    Byɔ  �          A^�H��{@�(�A"{B;  C���{@ӅAQ�B,�
CQ�                                    By�:  
�          AZ�\��=q@��\A!B?(�C}q��=q@��A�B/��C�                                    By��  
�          A]������@��
A33B8�C!H����@��HAp�B*  C�                                     By��  
�          A^�\�G�@�(�Az�B,�HC���G�@ᙚA�BffC
�f                                    By,  "          Ag��	@˅A  B)\)C��	@��A�B{C��                                    By�  �          A|z��@��A(Q�B'�CE�@�A��B(�CY�                                    By!x  
�          A_�
�\)@�
=A  B,Q�C\�\)@���A�HB�RCǮ                                    By0  "          A^=q��p�@��A$��B@G�C�q��p�@��
A�\B0�HC8R                                    By>�  
�          AT������@���A�\B4z�C�R����@�=qA(�B%33Cٚ                                    ByMj            AR{�ᙚ@��
A
=B9=qCs3�ᙚ@љ�A�B*
=C	�                                    By\  "          Adz����@��
A/\)BI�CO\���@�
=A'\)B<�HC�H                                    Byj�  @          A^{�	�?�
=A,z�BL�\C*� �	�@!�A)G�BGG�C#�{                                    Byy\  "          A^�H�\)?�Q�A*�\BHG�C's3�\)@AG�A&�\BA�
C �H                                    By�  
�          A_\)�
�\@ ��A'\)BD�HC#�\�
�\@dz�A"=qB={C�{                                    By��  T          Alz��!p�?�A,��B;{C-\)�!p�@�A*{B7�C'E                                    By�N  �          Av�R�#�=�G�A8(�BA�\C3ff�#�?��A7
=B?��C,Ǯ                                    By��  �          Aup��&�R>�A2ffB;�C1xR�&�R?�{A0��B9Q�C+33                                    By  �          Ai�ff���A+
=B<��C7}q�ff?�\A+33B<�RC1�                                    By�@  �          Ae���=q�*�HA ��B1�CC��=q��{A$(�B6�RC=B�                                    By��  
�          Al���"=q�w�A ��B+=qCH���"=q�333A&ffB2�RCCp�                                    By�  �          Al���!����G�A��B!
=CN� �!����Q�A z�B*�HCI��                                    By�2  �          Ae�
=����A  B!�CO\)�
=����A�B+Q�CJ��                                    By�  
�          A_���\��
=A	B�CSB���\����A�RB#�HCN�f                                    By~  T          AR{��\)��p�@��A��
C`���\)����@�B�\C]޸                                    By)$  �          ATz�������H@���Ař�Ce� �������@�
=A�(�Cc��                                    By7�  "          ATz����
��@�\)A�(�Cg:����
��@ƸRA�Ce\)                                    ByFp  T          Ae���ff� ��@��HA�ffCf��ff��@�(�A�Cd!H                                    ByU  "          Ah  �(��{@У�A���C`���(��{@�RA���C^+�                                    Byc�  "          As������\@�
=A�
=Cb������@��A�Q�C`^�                                    Byrb  �          Aep���z��*=q@ÅA�{Ck5���z���R@�
=A�=qCiL�                                    By�  �          Aq���@��HA��
Ca���
Az�B
=C^#�                                    By��  �          A�{�/�
����A"{B�RCQ���/�
��z�A,(�B&33CM{                                    By�T  "          A�{�,����Q�A!��B�CQ
�,����(�A+
=B((�CLL�                                    By��  
(          Ay���'33����A%B&��CM�=�'33�x��A-��B0��CHk�                                    By��  �          Ay���(���7
=A2{B5�CC#��(�ÿ�\)A5�B;�C<��                                    By�F  �          Au��!����A6{B@�
C;�H�!��\)A7\)BB�RC4Ǯ                                    By��            A[���G�@�@�33B�
B����G�A33@��B��B�G�                                    By�  "          ApQ�@\��Ag�?��R@�(�B�8R@\��Ai��>\?��HB�p�                                    By�8  T          Am@�\)Aa�?��@�B��H@�\)A`�Ϳ�G��z=qB���                                    By�  �          Av�H@�ffAf�H>#�
?(�B��{@�ffAe��Q����
B�ff                                    By�  "          An�H@�(�AZff��  ���B�aH@�(�AV�R�0���*�HB��                                    By"*  "          Aq�>#�
A\(�@���A���B��3>#�
AeG�@�
=A�G�B��q                                    By0�  
Z          Ah��>L��AT  @��A�p�B�=q>L��A]�@�G�A��
B�Q�                                    By?v  
�          Ae�����\AH��@���A��
B�B����\AT  @�ffA�=qB�                                    ByN  "          Ab�H<��
AO33@�  A���B��
<��
AXQ�@�  A��B��)                                    By\�  
Z          Adz�>��AQG�@�z�A�z�B��R>��AZ=q@��
A�Q�B���                                    Bykh  
�          AlQ�#�
A]�@�z�A��B�W
�#�
Ad��@a�A]��B�L�                                    Byz  
�          Ap�ÿ:�HAT��@�
=A���B�Q�:�HA`Q�@��A��\B�                                      By��  
�          Al�ÿ�(�AN{@�{A�z�B��
��(�AY�@�p�A�{B�B�                                    By�Z  T          Aj�\���AK\)@�z�A�  B��Ϳ��AW33@�(�A�\)B�B�                                    By�   
�          A~ff>�=qA`  @�ffAٙ�B��
>�=qAk�
@���A��\B��                                    By��  
�          A��@�A|Q�@��A�
=B�B�@�A���@���A\��B���                                    By�L  "          A�ff@��A{\)@�{A��B��3@��A��
@e�AD��B�=q                                    By��  �          A���?�A��@��A�\)B�\?�A�Q�@~�RAT(�B��                                     By��  "          A�(�?��
A|z�@�(�A�  B��3?��
A��
@��A�z�B��                                    By�>  "          A��?�{Av�H@���A��B�\?�{A�p�@�p�A��B���                                    By��  
�          A��\?�\)Aq�@��
Aȏ\B��f?�\)A}��@���A���B��=                                    By�  
�          A�Q�@,��Ao�
@�ffA�B��\@,��Az=q@�z�A�Q�B�ff                                    By0  "          A�  >8Q�Adz�A
�RA�(�B��=>8Q�As33@�p�AîB���                                    By)�  T          A�33�xQ�Aap�A
=B33B��)�xQ�AqG�@�ffA��
B�W
                                    By8|  �          A
=�{A8��A%��B$��B�#��{AK�AffB	�B�                                      ByG"  "          Aw�
��\)@��AL��B_�B�p���\)A��A=BHB��                                   ByU�  
�          AqG����RA�A733BHz�B�
=���RA=qA%B0  B��                                   Bydn            AV�\���@�(�A6�HBj�C�)���@�\)A-��BX�C�f                                    Bys  
Z          A`Q��˅�_\)A?\)Bk�CP���˅� ��AD��Bv�CE��                                    By��  
�          Af{��p���{AABm��CC��p���p�AC�
Br  C7\                                    By�`  
�          Af�H���@���AD��Bl��CaH���@�\)A:{BYG�C��                                    By�  	�          Aip���33@��
A<Q�BW��B�33��33A(�A-��B@�B�k�                                    By��  "          Aw
=��ffA�A5�BA�B����ffA(��A"ffB'(�B�ff                                    By�R  	�          A}G�>8Q�AO�A��Bz�B�k�>8Q�A_�
@��
A�Q�B��=                                    By��  
�          A�{@W�Aip�@�A��
B���@W�AtQ�@�=qA�ffB��                                    Byٞ  T          A�ff@��Ar=q@&ffAG�B�\)@��Au��?333@�B��H                                    By�D  T          A��H@�\Adzΰ�
���B~�\@�\A`(��C33�0(�B|��                                    By��  
�          A}G�@��An{�J=q�7�B��R@��Aj�R�)���{B�.                                    By�  
�          A}p�@�G�Ai녿�����HB���@�G�AdQ��mp��X��B��
                                    By6  
�          Az�R@���Aa���a��P��B�k�@���AX���������B�                                    By"�  �          Axz�@�{AYG�����(�B�Ǯ@�{AN{�˅���B��                                     By1�  �          Au�@ǮAN�H��G����B�� @ǮABff��z���(�B{\)                                    By@(  �          At��@љ�AG
=������RBx��@љ�A9G�������{Br
=                                    ByN�  "          At��@�{AM���\)���RBv33@�{AB=q�\��  Bp�                                    By]t  "          Aup�@�(�AO�
�|���o
=Bu
=@�(�AF{���H���HBpG�                                    Byl  "          A{
=@θRA`(�����|��B�z�@θRA\  �7
=�*ffB��                                    Byz�  �          Av=q@�  AW\)������(�Bv��@�  AR=q�S33�F=qBtz�                                    By�f  �          At��@��AO33�o\)�c33Bt(�@��AE����z���p�Bo�\                                    By�  �          Av�RA�\AH���n{�_
=Bc�A�\A?���=q��G�B^�
                                    By��  "          AxQ�A	�AFff�n{�]G�B\�RA	�A<���������BWp�                                    By�X  	�          Ax  A�A<(��x���h��BNp�A�A2ff�������BHp�                                    By��  
�          Aw\)A�A9��p�����BN33A�A.�H������BGff                                    ByҤ  
Z          Au��A#�A-�aG��T��B:�
A#�A$������  B4                                    By�J  	�          Az=qA%p�A-G����R����B9G�A%p�A"=q�����(�B1��                                    By��  �          A��A3\)A#
=��Q����B){A3\)Aff��=q��\)B��                                    By��  
�          A�{AH(�A������o�BffAH(�A
ff���
���B
�                                    By<  	�          A�
=A?\)A�H�`���K�B�HA?\)A������z�Bff                                    By�  
�          A���A=p�A�����
��33B\)A=p�A�R��(��ͅB�H                                    By*�  �          A�\)A<Q�A"�R��������B#ffA<Q�A�����
��33BG�                                    By9.  �          A�=qA>�HA  �߮��
=B�A>�HA��\)��z�Bp�                                    ByG�  �          A���A?
=A��ٙ����BQ�A?
=A������(�B=q                                    ByVz  
�          A��RA<��A!p������B"�A<��A{�Q��33B��                                    Bye   
�          Ab�\A'�@�p����H��=qB	�A'�@�\)��Q���(�A�                                      Bys�  
�          A=A�R@=q��{��Aq�A�R?�(���ff�=qA                                    By�l  T          A(�@�{>Ǯ�����(�@E�@�{���R����Q�C��H                                    By�  
�          A`  A2ff?�ff�ff�33A
=A2ff?(������R@XQ�                                    By��  T          As�
A5�&ff�"�R�'�C�]qA5��
���#\)C�޸                                    By�^  
�          A��\A<�Ϳ��R�,(��(�C�g�A<���>{�'33�"�\C��                                    By�  "          A{�A1G������1�3�C��HA1G��H���,z��,ffC�{                                    By˪  �          Atz�A2�H��  �%G��*=qC�ФA2�H�,(�� ���$��C�>�                                    By�P  T          A}A6ff���H�-��,Q�C�  A6ff�]p��'
=�$�
C���                                    By��  �          A~{A3��@  �-��+�C���A3���  �$���!��C��                                    By��  "          A}G�A*�R��33�-�-33C��A*�R��33�"{��C���                                    ByB  
�          Au�A)���\���'�
�-
=C�  A)������=q�!
=C���                                    By�  
�          A�z�A6=q�2�\�.ff�+�C�)A6=q��=q�&=q�!�HC���                                    By#�  T          A�  A5G��ff�6ff�2Q�C�&fA5G��}p��/
=�)z�C�^�                                    By24  "          A�Q�A.�R�E�3��2��C��A.�R��p��*ff�'�\C�j=                                    By@�  r          A�{A=q�ָR�4z��/\)C���A=q�Q��#
=��C���                                    ByO�  �          A�33@�
=�<  �?��)  C�u�@�
=�U��"=q�
�
C��                                    By^&  "          A�\)@��\�<���0���"
=C�*=@��\�TQ��\)�33C��                                    Byl�  T          A�  @�  �<Q��*=q�ffC���@�  �S33�����(�C�:�                                    By{r  �          A�=q@Ϯ�8���*�R�G�C���@Ϯ�P  ������HC�Ff                                    By�  
�          A���@�{�2{�:�R�+  C��@�{�K��ff��C���                                    By��  
�          A�  @���=���B=q�.�C��H@���X(��$  ��RC�k�                                    By�d  
�          A�ff@
=�Hz��K�
�3�HC�W
@
=�dz��+�
�G�C��3                                    By�
  �          A��\@�(��=q�9G��>��C�*=@�(��(Q��"{�!\)C�3                                    Byİ  
�          Am@��\��{�#
=�-G�C�:�@��\�{�\)�z�C��{                                    By�V  �          Ao33@���33� z��(�HC�@����
=�\)C�Ф                                    By��  @          Ak�@�{�  �)��9
=C��@�{�  �(��(�C��\                                    By�  
�          Ao�
@�\)�\)�)���4�C��=@�\)�+33�p��\)C��                                    By�H  
�          Ar�R@���z��'�
�/  C���@���0(���R��C�R                                    By�  "          Ax��@����!��-��1{C�c�@����9����\��HC��                                    By�  "          Ax��@�p����,Q��1G�C�@ @�p��6ff�=q�ffC���                                    By+:  �          A��\@��z��9�5C���@��733�����C��3                                    By9�  
�          A��H@�33���D���;
=C��f@�33�((��,���=qC���                                    ByH�  �          A��@��p��C�
�8ffC��f@��1�*ff�Q�C�l�                                    ByW,  r          A��R@�  �#��>�\�4�\C�33@�  �?33�"�H�33C�T{                                    Bye�  �          A���@��R�%�A��6C�o\@��R�A��%��
C��q                                    Bytx  
�          A��@��R�1G��:�\�-\)C��@��R�L(��z���C���                                    By�  	�          A�p�@���4z��8���+��C�/\@���O33�{�
C���                                    By��  
�          A�  @�  �8���,z��#G�C�/\@�  �Q���G���C��R                                    By�j  �          A�ff@����9�+��"{C�0�@����R�\�(�� (�C��)                                    By�            A�\)@أ�����9��1(�C�O\@أ��8Q���R�ffC�8R                                    By��  "          A��\@�=q�Q��Bff�9(�C��f@�=q�-p��(����RC��                                    By�\  �          A�(�@��
����C33�?�C��@��
�.=q�)��� (�C�(�                                    By�  "          A�@�����AG��C�
C��)@���
�*�\�&�HC���                                    By�  
n          Ax��@�G��4���  ��
C��@�G��G��ʏ\�ɅC�H                                    By�N  
�          Ah��?k��d(������G�C��R?k��f{>�(�?�p�C��{                                    By�  "          Ao33?�
=�c\)���
��  C�޸?�
=�k�� ����\)C��                                    By�  	�          Ap  @\)�`����(����
C�f@\)�i��\�Q�C���                                    By$@  
�          Al(�@n�R�5p��(����C��@n�R�I��љ���=qC�C�                                    By2�  
�          Al��@|���$Q����'�C���@|���<  � ���
=C�H�                                    ByA�  T          ApQ�@[��#��'
=�033C�AH@[��<���	���C�
                                    ByP2  T          A}G�@(���f�H���R���HC�,�@(���s��`  �LQ�C��=                                    By^�  
�          A�
=?���k33��p���C�.?���x���y���`  C�H                                    Bym~  
�          A}@33�ap��ۅ��  C�%@33�pz�����=qC��                                    By|$  
�          A�
@P���V�R��p���RC���@P���h�����\��  C�Q�                                    By��  
�          A�p�@7
=�bff�G����C��{@7
=�u����33��Q�C�E                                    By�p  
Z          A���@�=q�A���\�Q�C��)@�=q�Y�����33C��                                    By�  
�          A�(�@θR�333�#
=���C�  @θR�L  ��\�홚C�p�                                    By��  T          A�z�@��1G�� z��
=C�H@��I�� Q����HC�b�                                    By�b  T          A�(�@�\)�&=q�2{�$  C�޸@�\)�B{�\)�p�C��{                                    By�  T          A��H@�R�,  �+33�\)C��@�R�F�\����(�C��                                    By�  
(          A�Q�@�  �,���-����C�` @�  �H  ����p�C�z�                                    By�T  
�          A��R@�
=�3��&�R�  C���@�
=�M���p����C�>�                                    By��            A}@���G���{���C�@���Y����(�C��                                    By�  
�          A�=q@k��Y������Q�C��\@k��k�������\)C�f                                    ByF  
�          A��@C�
�^�H����{C�1�@C�
�p(�������  C���                                    By+�  
�          A�z�@����B�H����C���@����W���=q��z�C���                                    By:�  �          A�p�@�Q��3����=qC�R@�Q��L��������C��R                                    ByI8  "          A�Q�@���%��"ff�
=C��@���?
=�
=���C��                                    ByW�  T          A���@У��,  �!�G�C���@У��E����{C��                                    Byf�  "          A}p�@љ��'
=�=q�  C�@љ��@Q���z���C�K�                                    Byu*  
n          Az�H@������33�{C��@���-��������
C��                                    By��  |          A}�@�Q��*=q���C�5�@�Q��B�\��{���C��=                                    By�v  
F          Aw\)@j�H�W
=��  ��Q�C��@j�H�d���`���U�C�5�                                    By�  
�          Ar�\@�G��G33��  �֏\C��f@�G��W\)�����G�C���                                    By��  
�          Aj�HA�\��=q�ff�+�
C��A�\���(���C��\                                    By�h  @          Ak�
@�Q��!�Q��G�C�b�@�Q��7�����ӅC��f                                    By�  
Z          An{@\�
=���\)C�@\�4  ������RC�5�                                    By۴  	�          Ao�@��R�$(��G��{C�@��R�<z���=q��G�C�p�                                    By�Z  
�          A_�@���{��\�33C�W
@���+33�����
=C���                                    By�   
�          A]�@�p��$  �����C��{@�p��6�H������C��f                                    By�  �          Afff?�=q�Fff��z���=qC��?�=q�X  �����=qC���                                    ByL  
�          AZ�H?�33�>�R��z���(�C�� ?�33�N�H���H���
C���                                    By$�  
(          AW33��R�HQ����
����C��{��R�P�ÿ޸R��\)C�(�                                    By3�  	�          AJ{���9G�������RC�῕�C�
�(��3�C�B�                                    ByB>  
Z          APz��U��<�������=qC~E�U��E녿����\)C~�3                                    ByP�  	�          AU���.{�>�R�����
=C���.{�L���e�z=qC��H                                    By_�  
�          A[�
>.{�<Q���z���RC�k�>.{�M�������z�C�b�                                    Byn0  �          AY?}p��3����
=C��f?}p��F{��ff���
C�K�                                    By|�  
�          AT  @QG��3
=���
�݅C�'�@QG��B=q�y����  C���                                    By�|  T          AV�R@QG��0(���(���p�C�Ff@QG��A������C��\                                    By�"  	�          A^�R@5��EG���\)��
=C�y�@5��S
=�R�\�[�C��                                    By��            A]G�@�z��&ff��ff�z�C���@�z��;����\���HC���                                    By�n  
�          A_33@����(��(  �G��C�3@������Q��/��C�)                                    By�  "          Af�\@���p��0���HffC�@���{���+�C���                                    ByԺ  	�          Ai�@ۅ��z��2=q�I��C�{@ۅ��\�(��)�
C��                                    By�`  h          Ak\)@���p��%��5(�C�f@���*�R��R���C��\                                    By�  
�          Aj�R@�(��������&33C��3@�(��
=��Q�� �C�K�                                    By �  �          Ak
=@��
�������RC�q@��
�-���  ���C�!H                                    ByR  T          Aj�R@У������H� p�C���@У��,(���\)��C��q                                    By�  @          Am�@P���<����\�	��C���@P���S���{����C��=                                    By,�  
�          Alz�@`  �?33��(��Q�C�*=@`  �T(���z����C�c�                                    By;D  �          An�\@0  �U�������  C���@0  �d  �Tz��N{C�w
                                    ByI�  T          Ap  @=p��]p���\)��33C��@=p��h���	���(�C���                                    ByX�  �          Ap��?�  �X���\��z�C�&f?�  �g��Q��K\)C��3                                    Byg6  "          Ao\)��{�[
=���R���RC����{�h���8Q��2�RC���                                    Byu�  
�          AlQ�@G��1���
=��C���@G��K\)��G����C��                                    By��  |          Ah(�?���D����ff���
C��\?���X�����H��Q�C�u�                                    By�(  
�          Ah��@�33�(��"=q�2��C��{@�33�)���H�
{C�^�                                    By��  
(          Al��<#�
�C���{���HC��<#�
�U���H��  C��                                    By�t  T          Aq����(��O�
�Fff�>�\Cr)��(��U����녿���Cr�                                     By�  "          Atz�����V{��33��p�CrQ�����W�
?Y��@L��Cr��                                    By��  
�          Am����
=�PQ��$z�� (�Cs�\��
=�TQ�>�?�\Ct                                      By�f  "          Adz����H�B�\�e�h��Cq�\���H�J=q���
����Crz�                                    By�  T          A`���\�E�*=q�.�HCs�=�\�J=q���
���RCtL�                                    By��  T          AlQ���p��D�ÿ���陚Cm:���p��F�H?0��@+�Cm}q                                   ByX  �          Ag33��
=�L(��:�H�8Q�Cr5���
=�J{?�@�(�Cq��                                   By�  
�          Ah�����J�\>\)?\)Cpp����E@0  A.�\Co�
                                    By%�  	`          Ad  ��z��Ep������Co�R��z��A��@(�A�CoxR                                    By4J  
�          Adz����B�H������Cnٚ���@��?���@��Cn��                                    ByB�  
�          Af=q��33�@�׿
=q�
�HCl����33�>=q?�p�@��Cl�\                                    ByQ�  �          Ac�
�	�3���ff����Ch���	�1�?�z�@�  Ch#�                                    By`<  
�          Ad����z��G�
�0���1�Cr  ��z��E��?���@�p�Cq�                                     Byn�  T          AY��  �2=q��ff��33Ck+���  �/�
?�z�A ��Cj�=                                    By}�  
�          Ag\)�����K
=�W
=�Q�CqxR�����G33@�RA{Cq�                                    By�.  "          Ai����\)�B�H>Ǯ?��Cl���\)�=�@=p�A;�
Ck��                                    By��  "          Ak��{�:{?�33@�{Ch���{�0(�@��\A��HCg�                                    By�z  T          Ah����1�@��AffCf���%��@�Q�A�ffCd�H                                    By�   �          Ag���R�4(�?���@�\)Cg�)��R�)@�{A�z�Ce�                                    By��  �          Aj�R�
=�4z�?�{@���Cf�{�
=�)�@��A�=qCe!H                                    By�l  T          Ag�����-p�@   @��Ce\)����"�\@�G�A�ffCc��                                    By�  
�          Ah���{�'33@HQ�AG
=CcQ��{�Q�@�=qA�p�C`�                                    By�  "          Al(��z��'�
@\��AXQ�Cc��z��  @�z�A�C`&f                                    By^  �          Ah�����"=q@j=qAhz�Ca����@���A���C^ٚ                                    By  "          Afff���#
=@��A�33Cc�������@�{AˮC`33                                    By�  
�          Ac33�ff�\)@p��Au��Ca0��ff�
�\@�G�A��\C]�f                                    By-P  
(          Ad���p���@���A���C_���p����@ǮA���C\&f                                    By;�  �          Ad���G��(�@�z�A�z�C`��G����@�(�A��HC\8R                                    ByJ�  �          A[
=��G�@�(�A��C\+����33@�33A���CW�                                     ByYB  
(          A`(��=q�  @��\A��C^O\�=q���
@�
=A�(�CZ��                                    Byg�  "          A_�
��\�&�H?�{@�\)Ce� ��\�(�@�(�A��HCc��                                    Byv�  
�          A[���'33>���?�z�Ce����!��@-p�A6�RCd�R                                    By�4  
Z          Ab�\���'
=?�{@��HCd�=���(�@�z�A��Cb��                                    By��  "          Abff�
=�.�H>Ǯ?���Cf���
=�)�@5�A8��Ce�                                    By��  
(          A]G�����33�W
=�aG�Cb&f����  @�A33Ca��                                    By�&  T          AXz��=q��������z�Ca�R�=q�\)?�\)@�  Ca��                                    By��  T          AW�
�%������\���C[=q�%��z�?.{@9��C[u�                                    By�r  
�          AS�
�;�������R��=qCO\�;�����?��@��
CN�R                                    By�  T          AP(��E��^{?�{@��
CC��E��AG�@ffA'�CA�3                                    By�  
(          AN�\���
�R������\Cc
=������0  �F�HCe�=                                    By�d  �          AJ�H���
�
=��\)�ffCpٚ���
�'���ff����Cs�3                                    By	
  	�          AH����{�G���=q�
��Cr���{�'\)��G����Cuٚ                                    By�  
�          AL�Ϳ�(��33� (��!��C��3��(��,�����ڸRC��3                                    By&V  
�          AZ{@!�����+33�P�C��H@!��"�R�	G��G�C���                                    By4�  
�          Ab�R@�����&�H�>\)C��@���$(��z��z�C�                                      ByC�  |          Ae@�
=���H� ���3=qC���@�
=���{�
=C�                                    ByRH  "          Aa�@������$  �;p�C��@����{��  C��                                    By`�  @          AUG�@�����\�ff�<G�C�H@����=q��\)���C��R                                    Byo�  �          A/�
@���أ���  �.(�C�}q@���p����R���HC��q                                    By~:  "          A1������
��=q���HC�G����  �K�����C��R                                    By��  �          A(Q��p���=q����U�CO��p���=q?��@E�CO�                                    By��  
�          A#������ ���l(�����Cm�f�����
=��
=��Co�
                                    By�,  
(          A)����z��	p��)���j�RCm���z����
=q�=p�Co
=                                    By��  
�          AH�׿�Q��33�ff�0�HC�+���Q��ff��
=����C�&f                                    By�x            AZ�R� ����\�"�H�F��C�H�� ���)����(���C���                                    By�  �          Aj{��\)��p��)��Cv=q��\)�:{�����ffCz                                      By��  
�          Ap������9���G����Cs:�����Qp�������Q�Cu޸                                    By�j  
�          Atz�����EG���(���Q�Ct�����Y��n{�ap�Cv��                                    By  �          Atz���Q��?�
��=q���CsaH��Q��Vff��
=��(�Cuٚ                                    By�  T          Aw
=��z��Bff� (���  Cw\��z��[����\���CyxR                                    By\  �          Au���
=�9G��	��Cu�q��
=�U���G���p�Cx�)                                    By.  "          Aup����
�6�R���ffCw�����
�Tz�������G�Cz�\                                    By<�  
�          As�
�u��<�����C|  �u��YG�����\)C~@                                     ByKN  
�          Ar�\�[��;
=����Q�C}���[��X�������G�C�\                                    ByY�  �          Ao�
�%��8(�����C����%��V�R���
����C���                                    Byh�  
�          Al���Y���+
=�\)�#z�C|\)�Y���L  �����ٙ�C�                                    Byw@  
(          Ao��hQ��3��\)��
C|{�hQ��R�\�����=qC~��                                    By��  "          Aq��L(��>�R�
�\�\)C��L(��[
=��ff���C�s3                                    By��  �          Al(��L���4z��\)�z�C~#��L���QG���z����
C��                                    By�2  �          Ac33���ff����1��Cun���4��������\Cy��                                    By��  �          Ao
=�����3\)����{Cy+������P����\)���C{��                                    By�~  T          Ap����=q�3��
=��RCw�{��=q�Q����33����Cz                                    By�$  T          Ap  ��
=�5���H��RCvc���
=�Q������CyB�                                    By��  T          An�H��\)�*�R�=q���Co����\)�F�H��{����Csff                                    By�p  "          Amp����H�.�\��R��Cq�{���H�I���������Cu&f                                    By�  
�          Aq����<  ��p��ڸRCqL����Q��n�R�e�Cs�H                                    By 	�  T          An�H�޸R�@����ff��33Co���޸R�P���{���Cq�f                                    By b  T          Am����\)�Ap���������Cp����\)�P���	���G�Cr�q                                    By '  
(          Ak���z��B�\��\)���Ct(���z��S��(��G�Cu�q                                    By 5�  �          AX���Vff�0�������  C}#��Vff�F{�g��|  C~޸                                    By DT  �          AK
=�#�
�(������ffC�B��#�
�@  ���\��(�C�xR                                    By R�  
�          ADz�?#�
�"�R�ڏ\�\)C��=?#�
�9G��\)��C���                                    By a�  T          AB=q?��
�#
=�θR� �RC���?��
�8(��g�����C�.                                    By pF  �          A>�R?����'\)��
=�ڣ�C�U�?����8Q��%��HQ�C���                                    By ~�  
Z          AA��?��
�(z�������C�  ?��
�:ff�3�
�W\)C���                                    By ��  �          AB�R@��%���p����HC�q�@��8���A��fffC��                                    By �8  T          AD(�@S�
���ə����C�0�@S�
�4Q��^�R��(�C�1�                                    By ��  T          AC
=@(Q��&{���\���C��@(Q��8���:�H�]C�n                                    By ��  T          AG�
@0���*�R������  C�@ @0���=��333�P(�C���                                    By �*  T          AL��@\(��#������ffC�H�@\(��:�H�~�R����C�1�                                    By ��  T          AMp�@^�R�)G���p��C�R@^�R�>=q�Z=q�u�C�%                                    By �v  �          APz�@|���'
=�������C�\)@|���=��j�H��\)C�:�                                    By �  �          ALz�@P  ����"�\�\33C���@P  �Q��{�#�C�+�                                    By!�  �          A0��@L(������z��\��C�E@L(���33����%��C�`                                     By!h  "          A5�@e���ff�  �QffC���@e������H�z�C��                                    By!   �          A1G�@Q����H����_��C��@Q���(������(�HC���                                    By!.�  �          A$��@����{�
�\�f�\C��@����\�����-  C�Q�                                    By!=Z  �          A8��@j�H�����H�[�C���@j�H��(���33�$��C�~�                                    By!L   
�          A=@b�\��Q�����G�C��{@b�\�
=�����RC��                                    By!Z�  �          A7�
@�녿��
=�Wp�C�� @���vff���@Q�C��                                    By!iL  @          A&{@���*�H�
=�e  C���@����p������>�C�y�                                    By!w�  �          A)�@�
=�1G���H�j  C���@�
=���\��33�B�C�p�                                    By!��  "          A)�@\(��=p���C���@\(���p�����Q33C�/\                                    By!�>  T          A+33@e�k��
=\C��{@e�e����u��C���                                    By!��  �          A,��@?\)�����%��\C�"�@?\)�Fff���C��{                                    By!��  �          A-�@{>��)��)ADz�@{�(��%����C�&f                                    By!�0  "          A+�
?z�H?L���)��¦p�BQ�?z�H����&�\�{C�E                                    By!��  
�          A[
=A4(����������C�o\A4(���z����H��{C�8R                                    By!�|  �          AM�A&�R��p���33��\C��HA&�R�r�\�������HC��                                    By!�"  �          AB�HA+33�ٙ���(���33C�|)A+33�G���ff����C��                                     By!��  �          AG\)A$(������(���ffC��A$(���p��x�����C��                                    By"
n  �          A=�A���p���(���\)C�HA����1G��XQ�C�Ǯ                                    By"  �          A8  A����H��(���33C�{A���=q�`����
=C�Ff                                    By"'�  �          A1@�\)�9����z��'\)C���@�\)���H��G��
\)C��R                                    By"6`  T          A(  @��R?=p��  �[
=@���@��R�����V
=C�,�                                    By"E  
�          A��@ҏ\��  �ə��+�C��=@ҏ\�\�������33C�+�                                    By"S�  �          Az�@��\�8Q��ҏ\�A  C�~�@��\�"�\���
�/�C�>�                                    By"bR  T          A.{@��
���� Q��C�RC��3@��
�_\)��\�.Q�C���                                    By"p�  
n          A�
@ȣ�����z��<�HC��3@ȣ���Q��ȣ�� Q�C��3                                    By"�  
�          AQ��A��(��
=q�%�
C�~�A������
�=qC��                                    By"�D  
�          AR�HA\)�Z=q�z��-{C��3A\)���H��{��HC�xR                                    By"��  T          AMAz���=q���$C��\Az�����޸R��C�%                                    By"��  T          AG�
A�
���\����'{C�!HA�
�У���\)���C�Ф                                    By"�6  
�          AH  A(������H�#�RC��\A(���ff��G�����C�y�                                    By"��  
Z          AJ{A�\�p  ����
��C�eA�\��z���G����HC���                                    By"ׂ  "          AO�AG��l(������ffC��{AG���\)�ȣ�����C�޸                                    By"�(  T          AH��A!�3�
�����ffC�=qA!��z�����z�C��                                    By"��  �          AI�A����H�����{C�A����
��z����
C��q                                    By#t  �          AB�\@��H������'=qC���@��H� Q���G���\C�<)                                    By#  
�          AB{@�Q���G��=q�)�RC��R@�Q���ff����� =qC��)                                    By# �  
�          A8��A33�i������\C�HA33���H���
����C�ٚ                                    By#/f  �          A9��A  ��G���  ��\C�FfA  ���\��������C��                                    By#>  
(          A8z�A���QG���z��
=C�ФA����=q�������C��                                    By#L�  �          A:ffA���A���(��{C�� A�����\��ff���
C��3                                    By#[X  	�          A5p�@�\)�Q�������=qC���@�\)�33�(Q��XQ�C��                                    By#i�  
�          A4��@���=q��(���33C�<)@������u��=qC�*=                                    By#x�  |          A7�@��������  ���C�XR@��������  ���C�#�                                    By#�J  "          AC33Ap��J�H��(��z�C��Ap���(���(���33C���                                    By#��  T          A@Q�A��{��  ��
C���A���  ��ff����C���                                    By#��  
�          A>�\A�\�U��(��z�C���A�\�������\��RC��                                     By#�<  "          A<��A����=q��
=�G�C�H�A����Q���G���z�C��)                                    By#��  T          A;
=Aff��33��33�C�H�Aff����p���C��                                     By#Ј  
P          A7
=@���
�H����\C���@����\�&ff�S�C�.                                    By#�.  
Z          A6{@�{�33��Q����
C�(�@�{����>{�r�RC�ff                                    By#��  �          A7�
@�33�������RC��\@�33�
ff�����C���                                    By#�z  
�          A<��@����(��߮�
=C�@���ff������(�C���                                    By$   T          A<��@�\)��  ��p���\C�S3@�\)�  ��p����
C�XR                                    By$�  "          A;�@�
=����������C�b�@�
=�Q���ff��Q�C�%                                    By$(l  
�          A1p�@�
=��\)��ff��C��@�
=�{��{��(�C��R                                    By$7  "          A(Q�@�(���p���=q�=qC�l�@�(��33�qG���{C��H                                    By$E�  
(          A%��@��R���H�\�=qC��{@��R�����|�����C�aH                                    By$T^  
�          A'33@�
=�����ff���C�s3@�
=��  �������C�8R                                    By$c  
Z          A&�R@�ff������ff��RC�
=@�ff���R������{C���                                    By$q�  T          A'�@�z����
���H�Q�C��@�z����vff����C���                                    By$�P  
�          A(��@�
=��=q��ff�	p�C���@�
=���j=q��ffC��=                                    By$��  "          A'�
@�Q��θR�׮� {C�33@�Q��\)����=qC�                                    By$��  
Z          A)�@��R��=q�׮���C���@��R�G���
=�ǅC�j=                                    By$�B  	.          A*�H@�ff�������!�\C���@�ff��
���H��{C�|)                                    By$��  "          A,(�@�����33��=q��C��f@����=q��ff���
C�z�                                    By$Ɏ  �          A1G�@�(�������z�� \)C���@�(���
��  ���C��                                    By$�4  
�          A4(�@u���Q����%��C���@u��
=����Ǚ�C�                                    By$��  "          A4z�@����z������(�C��f@����������C�                                    By$��  �          A6{@|����G���ff�$C��{@|���������  C�O\                                    By%&  
�          A8��@HQ���{� Q��/��C�{@HQ������
=���C���                                    By%�  T          A9�@G
=������7\)C�Y�@G
=�ff������C���                                    By%!r  
�          A:{@3�
��ff�33�:��C�XR@3�
����ff����C��                                    By%0  T          A9@
�H��������6�HC�Ǯ@
�H� (���{�߮C��                                    By%>�  �          A;
=@&ff� ��� Q��-C��@&ff�"�H���\�Σ�C�+�                                    By%Md  T          A<  @�R�   ���2{C��)@�R�"�H�����֏\C��
                                    By%\
  "          A=G�@������
�1ffC�J=@���$(�������z�C��R                                    By%j�  
Z          AA�@G������
��RC�f@G��0Q����
���C��3                                    By%yV  �          A@z�@���H���R�z�C�Q�@��.�\�����33C��                                    By%��  �          A@  @3�
�
=��Q��!z�C��R@3�
�+33�������C�\)                                    By%��  T          A?
=@A���H���
�%=qC���@A��(  �������C�                                    By%�H  �          A=@G
=����
=�(�C�^�@G
=�%G���{���HC�b�                                    By%��  "          A=@S33���
�=q�5�C��R@S33�ff��Q����HC�9�                                    By%  T          A:�H@mp���Q����0z�C�"�@mp��\)��G����
C�q�                                    By%�:  �          A=G�@,������33�6�C���@,���!��  ���
C�z�                                    By%��  "          A:�\@Dz�������7G�C�)@Dz������Q���\)C���                                    By%�  �          A8��@E�����\�333C�  @E��G���������C��\                                    By%�,  �          A8��@ff�����\)�5Q�C�l�@ff�   ������33C���                                    By&�  
�          A6�\@
=q��z���{�0z�C���@
=q� z����R��z�C�{                                    By&x  
Z          A7�?��R��������9�RC�N?��R��H��z���(�C��                                    By&)  	�          A6�R?��R��{�
=�>z�C�y�?��R�Q���=q�陚C���                                    By&7�  	�          A8��@$z����p��9=qC�b�@$z������{��
=C�L�                                    By&Fj  
�          A6=q@'�����\�>�
C��@'��z������RC��                                    By&U  
�          A8Q�@�
�陚�
=�C�\C�� @�
�����H��33C��R                                    By&c�  �          A8��@|����{��{�&�C���@|���z���Q���ffC�                                      By&r\  
�          A;33@����\��{�*�C�8R@���(��������HC�h�                                    By&�  �          A8��@Vff��\�(��6C�K�@Vff�{������33C���                                    By&��  
�          A9G�@H�����H��R�3p�C�:�@H�������\)�֏\C��{                                    By&�N  
�          A9p�@7���=q����6�
C�` @7������H�ۮC�R                                    By&��  
�          A9@%����H�=q�2p�C��@%��!G���33���HC�+�                                    By&��  "          A2{@~{��
=����'�
C�k�@~{�(�����ĸRC��q                                    By&�@  "          A1�@�{��
=���#�RC��@�{�\)��{���C�<)                                    By&��  
�          A1@@  ��z����(�C���@@  ��R��\)��
=C���                                    By&�  "          A8z�@ ���   ����,�\C���@ ���"�\��Q��ÅC���                                    By&�2  
�          A9��@�33���H�*\)C��
@�%����p���{C�`                                     By'�  �          A9�@���p���ff�-�RC��=@���$z�������=qC�0�                                    By'~  "          A4Q�@�������\)�-�C��f@�� Q�������G�C���                                    By'"$  
�          A5�?����   �����.33C���?����"ff����33C�*=                                    By'0�  "          A6�R@�
�����Q��+p�C�%@�
�#���33��(�C��3                                    By'?p  
Z          A8��@�������\)�.�HC�` @���#33���\��{C��H                                    By'N  �          A9p�?����������5�C��?����#\)���H�У�C�ff                                    By'\�  �          A9p�?�
=���
����6�\C��f?�
=�#33������33C�]q                                    By'kb  T          A;33@<(��������2�
C�Z�@<(��!p����
���
C��                                    By'z  
�          A9p�@
=�����33�4Q�C�y�@
=�#\)��������C��
                                    By'��  
(          A4(�?\(�� (���33�1  C�\?\(��#33��p���ffC�g�                                    By'�T  T          A8z�>.{��R�{�3z�C���>.{�'
=��(���z�C�u�                                    By'��  
�          A4zὸQ�������H�9�C��=��Q��!p������
=C��q                                    By'��  
Z          A6{>�����(����?�HC�q>���� �����
����C��R                                    By'�F  
�          A4Q�>�33��{� Q��5  C�C�>�33�#33���H����C��)                                    By'��  �          A7�
?n{� z��
=�5��C�O\?n{�%p����R��Q�C��3                                    By'��  "          A8��@ff�����z��9�\C���@ff��\���R�أ�C��=                                    By'�8  �          A8��@w
=��\)��2��C�\@w
=��������  C���                                    By'��  "          A5@����(��   �3=qC�Ff@���������  C�|)                                    By(�  "          A5@�ff��(���=q�.{C��R@�ff�
�H������C�xR                                    By(*  "          A4  @�
=�������/�C���@�
=�\)��33�י�C�ٚ                                    By()�  
�          A5@��������ff�#�RC��H@���\)��=q����C�XR                                    By(8v  T          A5�@����G������.  C�P�@�����\)��  C���                                    By(G  "          A7
=@\)������;G�C�l�@\)�=q��
=��  C�˅                                    By(U�  �          A:�H@Z=q��33�	���>
=C��
@Z=q�{�����  C��H                                    By(dh  "          A:�H@Mp������
�;  C�ٚ@Mp��������  C��                                    By(s  
�          A;33@�R�������9(�C��@�R�#������z�C�(�                                    By(��  
�          A<��@4z����\��4��C��@4z��$  ���
�͙�C��3                                    By(�Z  "          A=G�@J�H�����(��233C��@J�H�"�\��G���z�C��=                                    By(�   
Z          A;
=@<����  ��
�3=qC�n@<���"{��Q���G�C��                                    By(��  �          A@��@,(�� ������4�
C�@ @,(��(z���ff��ffC�,�                                    By(�L  T          A@(�@0���ff����0�C�W
@0���(�����R�îC�P�                                    By(��  �          A?�@+��
=����/�RC��@+��)p�������p�C��                                    By(٘  
�          A?\)@$z�����R�,�RC��3@$z��*�R��\)���\C��                                    By(�>  
�          A?
=?������ff�,�\C�u�?����,����������C��                                    By(��  
�          A?\)@#33��� ���)33C�b�@#33�,(���������C���                                    By)�  T          AAG�@��
{��(�RC���@��/
=��G����RC�)                                    By)0  �          AAG�@  �����
�+�HC�^�@  �.�\����ffC��3                                    By)"�  
�          AB=q@��
=q�Q��+�HC���@��0(�����\)C�:�                                    By)1|  
�          AAp�?�z��Q���/(�C�P�?�z��.�H������G�C��3                                    By)@"  T          A<��?�z��G���\�.��C�w
?�z��+
=��p���{C�3                                    By)N�  �          A:{@�����{�,(�C�l�@��(  ��  ��  C��)                                    By)]n  "          A7�@'���
=���H�,�C��@'��#�
��\)���
C�.                                    By)l  "          A<Q�@.{��
��p��)�C�"�@.{�(Q����R��z�C�@                                     By)z�  "          A:�R?�(��
=���/{C��?�(��(�����
��ffC�N                                    By)�`  "          A8��?�z��p�� ���0{C��?�z��&�H�����  C�1�                                    By)�  T          A7�
?����ff��
=�/��C�\?����'\)������p�C���                                    By)��  T          A6ff?�33�
=���H�-�C��)?�33�'33��(�����C�!H                                    By)�R  �          A733?!G��Q����
�.
=C�+�?!G��(����(���{C��3                                    By)��  "          A7\)?J=q�  ����.�C�� ?J=q�(�������G�C�'�                                    By)Ҟ  �          A7\)?(������{�/�C�K�?(���(����{���\C�˅                                    By)�D  �          A:�\>�������z��*�HC�� >���,����������C�e                                    By)��  �          A:=q�����
�\�����'�C�箾����.{���
��C�!H                                    By)��  �          A:�\=#�
������&(�C�q=#�
�/
=�������RC�R                                    By*6  �          A:�R����������#�C�����/��|(���  C��                                    By*�  "          A:{<#�
�
�R��\)�&�HC�
=<#�
�.=q�������C��                                    By**�  "          A9��>��	p���G��(�
C�k�>��-G���z���\)C�U�                                    By*9(  �          A9G�����z����\�*�C������,����{���C�N                                    By*G�  "          A9��>u�����\�#�C�˅>u�/33�w
=��p�C���                                    By*Vt  �          A9p�?\)�
=���H�$=qC���?\)�-�y�����C�|)                                    By*e  �          A8��?L������(����C��
?L���/33�hQ���z�C��                                    By*s�  "          A7�
?(������z��C��?(���/��W���Q�C���                                    By*�f  T          A7\)>�ff������!�C�w
>�ff�-p��l������C�.                                    By*�  �          A6{>aG������  �$�
C��R>aG��+\)�u���C��3                                    By*��  
�          A5�>���
�\���!33C��>���,(��j=q��Q�C�@                                     By*�X  
(          A7�?��	���G��$p�C�?��,���u����C�g�                                    By*��  
�          A:=q?Tz��z���=q�"p�C��\?Tz��/33�s33���RC�'�                                    By*ˤ  �          A;
=?�
=�����G�� �C���?�
=�/��p����(�C���                                    By*�J  "          A:�\>�ff�Q����
��C�k�>�ff�1�aG����RC�'�                                    By*��  T          A9p�?(������Q��!��C��)?(��/
=�n�R��z�C��R                                    By*��  �          A9�?#�
�������"p�C�3?#�
�/��qG����C��=                                    By+<  
�          A8z�=�������
�  C�S3=����/\)�c�
��{C�C�                                    By+�  T          A7�
�u�z����({C�˅�u�,Q��~{��  C���                                    By+#�  �          A7
=<����ٙ��C�{<��/�
�:�H�n=qC��                                    By+2.  "          A5�>�����ff�
=C�q�>��/33�5��g33C�4{                                    By+@�  �          A3\)>�Q��(���\)�
=C�/\>�Q��+��Mp���G�C��R                                    By+Oz  �          A3�?��z�����!z�C���?��*{�dz���(�C���                                    By+^   T          A1�>W
=�33��
=�"  C��R>W
=�(���c33����C��3                                    By+l�  "          A1G�>�p��	������C�=q>�p��)��S�
����C�H                                    By+{l  
�          A1�>��
�
ff��{���C�>��
�)��L����=qC��)                                    By+�  �          A3
=�#�
�p���G��  C��ü#�
�-��*=q�\(�C��
                                    By+��  T          A3\)����=q��{�z�C������.=q�"�\�Q�C�                                    By+�^  �          A2{�!G��=q��G��	�C��!G��-p�����G�C�T{                                    By+�  �          A4z�(����(���C���(��.�H�.{�_
=C�j=                                    By+Ī  �          A1G���ff�z���Q��	��C��=��ff�+�����Hz�C��                                    By+�P  T          A0  �������\�z�C�Ǯ����*�H��R�:�HC�j=                                    By+��  �          A/\)��ff��
��{�	ffC��=��ff�*�\��D��C�/\                                    By+�  
�          A.�\��ff�ff���H���C��ῆff�*�H���H�%��C�1�                                    By+�B  �          A.�R��33��������ÅC��쿳33�-G�����HC�O\                                    By,�  
�          A,Q쿆ff�Q���G���C��쿆ff�(�׿����&�\C�*=                                    By,�  
�          A-��=p��
=q�����HC��=�=p��&�R�*=q�dQ�C���                                    By,+4  �          A,�Ϳ=p���������
=C����=p��'
=�:=q�xz�C���                                    By,9�  �          A+�
�\(��z���� (�C�G��\(��((��������C���                                    By,H�  "          A)G����������Q��
�\C��������#
=��
�IC���                                    By,W&  �          A+�
�ٙ��p����H���C��=�ٙ��&ff��\�/\)C�XR                                    By,e�  �          A*=q�޸R�����=q��C�=q�޸R�#��ff�L  C�(�                                    By,tr  T          A)�������	p�������\C�������#
=���>{C��)                                    By,�  
�          A(Q쿗
=�(����H�Q�C�Ϳ�
=�#
=�Q��P��C���                                    By,��  "          A*{�8Q��
ff��33�Q�C��)�8Q��%����J�RC���                                    By,�d  �          A,�׾aG��	G��Ϯ�p�C�B��aG��&�\�.{�i�C�c�                                    By,�
  �          A,�Ϳ����R���
���C��q����'�
� ���,��C��                                    By,��  "          A+���{�����z�C��΅{�&�R�ff�4��C�E                                    By,�V  �          A2=q����Q������\)C������,z�����K33C��
                                    By,��  
�          A4Q�B�\�
=�ڏ\�{C��3�B�\�-��7��k�
C�H                                    By,�  �          A4�ÿ�{�33��p��$(�C�Ff��{�*ff�fff��z�C�
=                                    By,�H  �          A/�
��
=�{��z��	�RC�7
��
=�(���  �?
=C�Ф                                    By-�  �          A.ff��33�z���p��C��ῳ33�)��p��'�
C�AH                                    By-�  �          A+
=�33����� p�C�xR�33�%����ffC�e                                    By-$:  
�          A+33���33��
=�G�C�
���#
=�{�V{C�=q                                    By-2�  
�          A,  ��(������ �HC��ÿ�(��!��Q���{C��q                                    By-A�  �          A,  ��  ���������
C�Ff��  �((���
=��C���                                    By-P,  "          A/�
��=q���p���C���=q�+33��
=�!�C���                                    By-^�  
�          A녿�\)��=q��\)���C(���\)�p��1G���(�C��                                    By-mx  �          A  ��z���=q������C~W
��z���H�(��|  C��)                                    By-|  	�          A�
�<(���33���R�$Q�CsT{�<(����>�R����Cx��                                    By-��  �          A
=q�l����ff��\)�0G�Ci:��l����z��e��
=Cq                                    By-�j  "          A�\�tz����\�׮�E��Cd���tz����
������
=Cp                                      By-�  T          A\)�z=q��33��  �A�Ch�\�z=q�������\)Cr�\                                    By-��  T          Aff�Tz���
=�θR�-�Cp���Tz�� Q��g���
=Cw�                                     By-�\  
�          A(��hQ���ff���*��Cn�H�hQ�����e��{Cu��                                    By-�  
�          A33�{����
��z��ffCnQ��{��(��J=q��z�Ct��                                    By-�  �          A���tz���z���(���Cq��tz��
=q�.{��=qCv(�                                    By-�N  
(          A!G��HQ���Q���
=�Cv���HQ���8Q����C{\                                    By-��  �          A&�\�Dz���  �����p�CwǮ�Dz��\)�Dz����C|�                                    By.�  "          A'33�I����\)�����
Cw&f�I���33�Fff��ffC{�\                                    By.@  
�          A(���J=q��\���
�ffCwff�J=q���G
=���C{�                                     By.+�  
Z          A$(��.{�����ƸR���Czs3�.{���,���u��C~                                      By.:�  T          A)��r�\���R�����z�Cs���r�\�(��(Q��f=qCxE                                    By.I2  �          A((��y����\)���� ��Cs��y������ ���0��Cw�                                     By.W�  "          A,z��~{����p��p�Cs���~{�  ����B{Cw�
                                    By.f~  �          A,Q��n�R��\)�Å�
�Ct���n�R��
�p��T��Cy\                                    By.u$  
Z          A0���x����\�����Ct���x����
�%��Xz�Cx�R                                    By.��  �          A4  �Q������
�33Cx�H�Q��&�\�.�R�`��C|�                                    By.�p  
�          A8(��(����H���
��C}�R�(���/33��R�333C�:�                                    By.�  "          A7
=������������C�����0�ÿ\��(�C��                                    By.��  T          A6ff�>�R�p���ff��{C|�q�>�R�/\)���H���HC~��                                    By.�b  "          A?
=�fff����  ��Cy��fff�5p���\)�ҏ\C|\)                                    By.�  "          A6�H�^{��
��33��G�Cyn�^{�,�Ϳ��H���C|33                                    By.ۮ  �          A7\)�[���������CyǮ�[��,�׿��
��p�C|c�                                    By.�T  T          A?\)�e��
��{��Q�CyǮ�e�4�Ϳ˅���
C|c�                                    By.��  T          A:�\�\(������  ��\Cz#��\(��0�׿��R���HC|��                                    By/�  �          A5�O\)����������HC{:��O\)�-������C}^�                                    By/F  
�          A;�
�p����
��ff��z�Cx���p���1��\)���RC{J=                                    By/$�  T          A<(��J�H� ����{�љ�C|� �J�H�4z�@  �j�HC~Q�                                    By/3�  �          A1��:=q�  ��p�����C|�3�:=q�*�R�:�H�p  C~��                                    By/B8  �          A2=q�a��
=���R��z�Cy��a��(�׿���7
=C{�                                     By/P�  T          A5���l������
=��=qCx��l���*�\�E��z=qCz�H                                    By/_�  "          A<(��`  � �������p�CzǮ�`  �3�
��R�A�C|�                                    By/n*  
�          A5�XQ��\)��=q����Cz���XQ��-G����+�C|�f                                    By/|�  �          A5p��H���z������ǮC|8R�H���-녾���
=C}��                                    By/�v  �          A?\)���R�*�R��\)��33C��ÿ��R�<Q쾔z῱�C�1�                                    By/�  T          AIp���p��7���������C�쾽p��Hz�        C�&f                                    By/��  �          AK33>B�\�<  ��������C�xR>B�\�J�H>�{?��C�p�                                    By/�h  T          AL  ���<z�������p�C��=���K\)>���?��RC���                                    By/�  
�          AM�=u�?
=�����p�C�'�=u�M�>Ǯ?�  C�%                                    By/Դ  �          AAp��k��2�R������33C�g��k��@��>���?�ffC�s3                                    By/�Z  
�          A9녾.{�.�\��G���ffC����.{�9?0��@Z=qC���                                    By/�   
�          AD  ���
�8  ���R����C��ͽ��
�C�?E�@g
=C��\                                    By0 �  
�          AG�    �;33��=q��{C�H    �G33?B�\@^�RC�H                                    By0L  
�          AJff��G��<����Q���33C�녾�G��I?!G�@6ffC��q                                    By0�  
�          A)p��W
=�p��xQ���Q�C�b��W
=�)�>��@\)C�l�                                    By0,�  �          A9������&=q��Q�����C��q�����6�R�#�
�L��C�H                                    By0;>  T          AD���#�
�-�����Ə\C�\)�#�
�@zᾮ{����C���                                    By0I�  
�          AB�\�Fff�)G����
��33C}��Fff�;�
��녿�C8R                                    By0X�  T          A@  �8���'�����Ǚ�C~�\�8���:{�Ǯ��\)C��                                    By0g0  �          A>�R�6ff�*�H������C��6ff�8��>�=q?�ffC��                                    By0u�  �          A<z��   �)p����R��ffC�Y��   �7�
>W
=?�G�C���                                    By0�|  "          A1G��H���z������ffC|5��H���)��>L��?��C}��                                    By0�"  �          A%�L(�����|(�����Cz�\�L(��=u>���C|�                                    By0��  
�          A{=#�
��\��ff��C�#�=#�
�p��
=�c\)C�q                                    By0�n  �          AQ�>\�����  �C�w
>\�\)�{�pz�C�.                                    By0�  "          A��>W
=��������33C���>W
=�ff�G��A��C��f                                    By0ͺ  
�          A�\>#�
��{���\���C��
>#�
�ff����S33C�z�                                    By0�`  
�          A{��=q����H��Q�C��)��=q�  �\)�J=qC��                                    By0�  
�          A   �u�{���R��=qC�녿u��H���*=qC�>�                                    By0��  
Z          A�H������tz�����C��H����p�>L��?�
=C��
                                    By1R  
�          A"�H��R�33�Dz���(�C�\)��R�{?�ff@��C���                                    By1�  T          A��
�H�p��(Q��{�C�G��
�H��?�ff@�ffC�z�                                    By1%�  !          A���\����5����Cn��\�
=?}p�@�\)C��                                    By14D  �          Az��\)�z��y����p�C��H��\)�{���
���HC��                                    By1B�  
�          A����33�\(����HC�Ǯ���p�>��@p�C�33                                    By1Q�  
(          AQ�Y�������Q���  C�,ͿY���\)�.{��G�C�p�                                    By1`6  
�          A�?���\���� C��R?���R�����
=qC��)                                    By1n�  "          A  �   �	G���������C�Q�   ���.{���C�z�                                    By1}�  �          A��?z�H�(��W
=��
=C�1�?z�H��?�\@G
=C��)                                    By1�(  	�          A��?u����.{��z�C�H?u��R?��R@�ffC��f                                    By1��  
�          A\)��=q�=q�ff�f�RC�#׾�=q���?˅A�C�'�                                    By1�t  
�          A���ff����.�R��(�C��\��ff�?��H@�C��)                                    By1�  
�          A=q��33�33�   �q�C�p���33��R?�p�A{C��f                                    By1��  	�          A\)���\���{�e�C�7
���\��
?�z�A�C�G�                                    By1�f  
�          Ap��E��
=�U�����C����E��Q�?!G�@p  C���                                    By1�  T          Az�?aG���  �����
��C�=q?aG��z��  �)C���                                    By1�  "          A33��ff�	��~�R�¸RC����ff�  ���Ϳ�C��H                                    By2X  
�          A��E�	p��.�R����Cz8R�E�33?��@ȣ�Cz��                                    By2�  
Z          A33�Tz��\)�&ff�yCy��Tz��(�?���@��
Cy�q                                    By2�  
�          AQ���ff�
ff��(��"{Ct
��ff�	��?��RA;�Cs�                                    By2-J  �          A�R�����
=��  �
�HCru��������@p�AMCr
=                                    By2;�  T          A
=��
=����Q��z�Cr�)��
=�
ff@33A=�Cr�f                                    By2J�  �          A����{�	���(��Cr����{�	p�?��A1G�Cr�f                                    By2Y<  T          A\)��(��(������Coff��(��G�@�AP  Cn�
                                    By2g�  �          A����H�G���z���Cm�=���H���H@G�AZ=qCm�                                    By2v�  T          A
=��ff��(����R�G�Cm�H��ff����?��A9�Cm��                                    By2�.  T          A�R��=q��(��ٙ��'33Cn�\��=q��z�?�Q�A&=qCn�\                                    By2��  �          A=q�����\�Q��d��Cg�H������?Y��@��Chٚ                                    By2�z  T          A�����
�����hQ���z�CX� ���
�ə�������HC]}q                                    By2�   �          A�R��z����H�.�R��C_� ��z���(�>k�?��Cb�                                    By2��  �          A$(��������
=�U��Ca&f������?B�\@�  Cb��                                    By2�l  
�          A#�
�љ������
=�   Ce�q�љ���z�?��A*�\Ce^�                                    By2�  "          A"�R��\)��\)���R��RCd
=��\)��?�p�A��Cc�
                                    By2�  
`          A"{����\���.{C`Y�����  @3�
A�
=C]�f                                    By2�^  
�          A   ��\)���;�\)��{Ca�R��\)��(�@,��Az�\C_��                                    By3	  T          A�\�����\���Ϳ
=Ca�R����Ϯ@5�A�33C_8R                                    By3�  
�          A#�
�#33�=q�Dz���(�C~���#33��?�=q@�33Cp�                                    By3&P  
�          A,z��p���
��(���ffC�C׿�p��)G�>�=q?�
=C��{                                    By34�  	�          A*=q��(��G���33���C����(��(Q�<#�
<�C�f                                    By3C�  T          A*{�Q�����z�H���C�
=�Q��%>���@	��C���                                    By3RB  �          A.{�
=�  ��Q���  C�1��
=�*=q>#�
?Tz�C��q                                    By3`�  
�          A9��ff�=q��z���33C�S3�ff�3�
�O\)����C��                                    By3o�  "          AA��S33�*�\��\)����C|�\�S33�:�\=�G�?�C~5�                                    By3~4  �          ADQ��}p��*�R�������Cy���}p��9��>�=q?�  C{#�                                    By3��  "          AA�����'\)�L(��up�Cs������-p�?�
=@�G�CtǮ                                    By3��  �          A@Q���\)�'��G��q�Ct����\)�-G�?�  @��CuT{                                    By3�&  �          A4Q��XQ�����G����RC{��XQ��,(�>B�\?s33C|�{                                    By3��  
�          AAG���ff�+\)�g
=����Cx�
��ff�3�
?�33@��RCy��                                    By3�r  
�          AG
=�����1��g
=��\)Cxٚ�����9�?�ff@���Cy��                                    By3�  "          AC���=q�0  �H���n{Cx����=q�4��?ٙ�A   Cy
                                    By3�  �          A7�����%G��*=q�UCw^�����((�?�\)AG�Cw��                                    By3�d  �          A6�H��  �"�\�*�H�Xz�Cv)��  �%��?��
A�Cv��                                    By4
  �          A3
=���\��\�:=q�p��CvaH���\�#�?��H@�  Cw�                                    By4�  �          A3��x����H�H����G�Cx�)�x���%p�?�G�@У�CyaH                                    By4V  
�          A4z��)���%�c33���HC�f�)���.ff?���@��RC�+�                                    By4-�  "          A9p��u�&�\�Tz����Cy� �u�-��?�ff@�
=Cz��                                    By4<�  �          A9p��b�\�)p��G��z{C{�=�b�\�.�H?Ǯ@��RC|{                                    By4KH  
�          A8  �fff�'��E��x��C{��fff�,��?�ff@�\)C{�
                                    By4Y�  T          A7�
�c33�'�
�C33�v�HC{G��c33�,��?�=q@�(�C{��                                    By4h�  �          A5��dz��%��7
=�j=qCz���dz��)�?ٙ�A
{C{h�                                    By4w:  "          A1����Q��Q��;��u��Cvn��Q��!��?�33@�Q�Cw!H                                    By4��  �          A6�R��p��!��;��m��Cvh���p��&�R?��@�  Cw                                    By4��  T          A8(���\)�%p��!G��H��CtB���\)��R@n{A��Cr�R                                    By4�,  
�          AD  ��=q�-p��%�C33Ct����=q�/
=@	��A!�Cu&f                                    By4��  �          AE���
=�.�\�5�T��Cu���
=�1��?���A�Cu��                                    By4�x  T          AC���33�,���   �=G�Ct�q��33�-�@{A'33Ct޸                                    By4�  T          AD  ��\)�,Q��$z��A�Ct)��\)�-�@��A ��CtO\                                    By4��  
�          AD����Q��,���#�
�@Q�Ct
=��Q��.=q@
�HA"ffCt8R                                    By4�j  �          AD����\)�,z��.�R�MG�Ct���\)�/
=@   ACtn                                    By4�  �          ADz����R�,z��/\)�N{Ct5����R�/33?��RAp�Ct�=                                    By5	�  T          AC�
�����,z��(���G�Ctz������.ff@�AQ�Ct��                                    By5\  �          AD(���ff�,(��0���P  Ct0���ff�.�H?��HA\)Ct��                                    By5'  
�          AC�����*�R�1G��QG�Cs�����-��?�AQ�Cs�                                    By55�  �          AC���Q��-��*�H�I�Cu.��Q��/33@z�AQ�Cup�                                    By5DN  T          AC33���H�,��� ���>{Ct�=���H�-�@p�A&�HCt�                                    By5R�  �          AB{��p��*�\��Q��33Cs���p��(  @*�HAL  Cr��                                    By5a�  
�          AD�����
�/���Q�Ct�����
�-��@*�HAHz�Ct�R                                    By5p@  �          AC�
��=q�/���\)���Cu8R��=q�,Q�@7
=AXz�Ct�=                                    By5~�  "          A;
=��\)�((������p�Cu���\)�%�@&ffAMCuz�                                    By5��  �          A;���(��(�ÿ�{���Cu33��(��$z�@;�Ag\)Ct�
                                    By5�2  �          A>�H���H�*�H�������Ct}q���H�%��@B�\Alz�Cs�=                                    By5��  �          A<(���G��(zΎ�����Ctk���G��"=q@J=qAy�Cs��                                    By5�~  "          A9������%G���{�أ�Ct������@C�
At��Cs0�                                    By5�$  �          A8������%���ٙ��33Ct�)����!�@0��A]�Ct\)                                    By5��  
�          A4(���
=����{��Cs���
=�  @,(�A\��Cr�R                                    By5�p  
�          A4Q������ff�����RCs
=�������@��ADz�Cr�=                                    By5�  
�          A5����R����Q���Cs�����R�{@Q�AB�HCsT{                                    By6�  
(          A4Q������ z���
�)G�Cu#�������@�\A<Q�Cu�                                    By6b  �          A2�H������Q��\)Cs(������R@#�
AS�
Cr��                                    By6   �          A0����  �����׮CtE��  �(�@:�HAv{Csp�                                    By6.�  �          A0(�������H�aG����HCq�q������R@N{A�ffCp�                                    By6=T  �          A"=q�qG���
��Q��=qCw@ �qG��Q�@�AdQ�Cv�                                     By6K�  �          Ap�@   ��ff��\)��z�C��\@   �Q쿧���C�+�                                    By6Z�  �          A{?�\)��p���  ��Q�C�\?�\)�Q�G����C�.                                    By6iF  �          A�׿������p  ��C�������ff�u��Q�C�z�                                    By6w�  �          Az�?k���R��(���p�C�)?k��
=�p����  C��{                                    By6��  �          A�
@=q���R��G���HC��@=q�\)��{�p�C�T{                                    By6�8  �          A Q�@�33����z��  C���@�33��
=�$z��|��C�Q�                                    By6��  �          A)��@�ff��  ��{�	=qC�xR@�ff��  �N�R��z�C���                                    By6��  �          A'\)@��R��\)���\���C���@��R�Q�޸R�C�j=                                    By6�*  �          A&ff@�Q�������z���Q�C�w
@�Q���ff��\�Lz�C�q                                    By6��  T          A%G�@�������ʏ\��C���@����(��{���C��H                                    By6�v  �          Ap�@���=p����
�"�C�>�@�����H����ޏ\C��3                                    By6�  �          A-G�@�������\)�.�HC�|)@����{��{��\)C��R                                    By6��  �          A-��@�=q�ə��ڏ\�p�C��H@�=q����g
=���C��f                                    By7
h  �          A%G�@������G���C�1�@������'��j�RC�&f                                    By7  �          A�\@�ff��Q������C�g�@�ff�ff����HQ�C��)                                    By7'�  �          AG���  ��
=�����8��Ch�3��  ���H?��ACi&f                                    By76Z  �          A=q��p����Ϳ����p�C_����p���Q�?��HA)G�C_(�                                    By7E   �          A������p�>���@(�CS^������@=qAk�CP)                                    By7S�  �          A   ����N{?�(�A{CG{����{@9��A��RCAh�                                    By7bL  �          A Q��\)�C�
@p�Ac�CF\)�\)��G�@`��A��C>�
                                    By7p�  �          Aff��\�n{@(�ALQ�CJ�f��\��R@a�A�33CC�=                                    By7�  �          AG����U@p�AP  CHk�����@X��A�CAQ�                                    By7�>  �          A���
��?���ACZ�\���
��\)@z=qA���CT�3                                    By7��  �          A��������?�33A�HC[� ������@r�\A�Q�CV)                                    By7��  �          A33��z���G�?��A33CT�{��z���Q�@VffA�  CN�R                                    By7�0  �          A=q����Q�?˅A  CR����j�H@Z�HA��RCL�{                                    By7��  �          A�
=�Tz�@�\AJ�HCIn�
=�
�H@N�RA���CBn                                    By7�|  �          Aff�
ff�(�@6ffA���CB0��
ff�O\)@`  A�p�C9^�                                    By7�"  �          A��  �޸R@333A�33C?@ �  ��(�@P��A��C6��                                    By7��  �          Aff�
=q�C33@z�AK�CGn�
=q��@H��A�ffC@��                                    By8n  �          A�R�
�H�W�?�{A��CI8R�
�H� ��@(Q�A��CD)                                    By8  �          A�
�33�l��?Tz�@���CK\�33�A�@�\A_33CG0�                                    By8 �  �          Az����r�\>��@5�CKxR���P��?��RAAG�CHu�                                    By8/`  �          AQ��(��]p�?���@��CI�\�(��'
=@(Q�A��\CD��                                    By8>  �          A(�����N{?�\)A��CH!H����G�@333A�\)CBu�                                    By8L�  �          Az��  �^{?�A��CI�H�  �%�@.{A���CDh�                                    By8[R  �          A����o\)?�G�@��HCK33���8Q�@,(�A��RCFL�                                    By8i�  �          A��z��p  ?�(�@陚CK��z��:=q@*=qA�=qCFT{                                    By8x�  �          A{�  �u?�@޸RCK���  �@��@(��A33CG�                                    By8�D  �          A(��\)�n�R?��\@���CJ���\)�>�R@{Aj�RCFff                                    By8��  �          A���33�z=q?u@�z�CK�)�33�J�H@\)Ak33CG}q                                    By8��  �          A!�  �r�\?�Q�A33CJG��  �7
=@7�A�=qCE.                                    By8�6  �          A33�
=�c�
?�ff@�ffCI�q�
=�-p�@)��A}��CD�H                                    By8��  �          A�R��\�\(�?��
Ap�CI#���\� ��@3�
A�
=CC�q                                    By8Ђ  �          A ���(��fff?��\@��CIE�(��0��@(��AtQ�CD��                                    By8�(  �          A$������mp�?�=q@�Q�CI8R����;�@ ��Aap�CE�                                    By8��  �          A"ff���c�
?G�@���CH�����;�@
�HAE�CE0�                                    By8�t  
�          A!��{�mp�?=p�@�p�CI���{�E�@��AH��CF33                                    By9  �          A"=q�Q���G�?!G�@c33CK�=�Q��[�@�RAK�CHG�                                    By9�  �          A!��z���  ?&ff@k�CKL��z��XQ�@�RAK�CH                                      By9(f  �          A Q������>�33@ ��CL@ ���fff@ ��A8��CI�\                                    By97  �          A ���\)����=u>�p�CK�\�\)�h��?��HAQ�CI��                                    By9E�  �          A ���
=��G��#�
�n{CK���
=�p  ?�G�A
{CJ33                                    By9TX  �          A���������ff�*=qCL޸���|(�?��\@�ffCK�                                    By9b�  �          A�R����{��Q��
=CM\���~�R?���A z�CK��                                    By9q�  T          A��{���þ��ÿ�33CO�{����?�ffA33CM��                                    By9�J  �          A
=����=q�8Q쿅�COff����  ?��HA{CM�3                                    By9��  �          A{�Q����׾8Q쿆ffCO=q�Q���ff?�
=A��CM�\                                    By9��  �          A�������H�\)�Q�CNE�������?�33A�\CL�\                                    By9�<  �          A���
���\���
��G�CO���
��
=?�ffA(z�CM�                                     By9��  T          AQ��
=��p�?
=q@J=qCN�3�
=�s�
@�\AY��CK��                                    By9Ɉ  �          A��
{��
=>\@�RCOaH�
{�{�@
�HAN=qCLu�                                    By9�.  �          A��\)���>�33@CNQ��\)�r�\@�AE��CK�                                    By9��  �          A�\�	���>�@0��CN���	�p  @�AQ�CK�                                    By9�z  �          A�H�	p���������33CQxR�	p����
?�G�@��CQ�                                    By:   �          A�
�(�������33�G�CR�3�(���{?E�@�{CS^�                                    By:�  �          A�R�p����?(�@g
=CR@ �p���ff@"�\As
=CN��                                    By:!l  T          A�
�	p�����?�@ECOǮ�	p��z�H@z�A\��CL��                                    By:0  �          A�R�
�H���>��@333CN��
�H�k�@	��AMCJ��                                    By:>�  �          AG��Q�����?!G�@k�CN�f�Q��p��@
=A^�\CK:�                                    By:M^  �          A!���
���?(��@o\)CN����
�u�@�HA^�RCK�                                    By:\  �          A$���=q��>�ff@�RCO!H�=q��33@�
AO�CL(�                                    By:j�  
�          A%�33��ff�\�Q�CR#��33���?�  A(�CP��                                    By:yP  �          A#�
�z���Q��(���CR���z����?޸RA(�CQ�)                                    By:��  �          A#\)���������,(�CR^�������?�33A��CQ(�                                    By:��  �          A#������=q��
=�CQ��������?�A�HCP��                                    By:�B  �          A#\)����33�
=�U�CR������?\A��CQ\                                    By:��  �          A$Q��������333�z�HCR�q������\?�(�A  CQ�3                                    By:  �          A$���33���R�\(�����CT#��33���H?�@�{CS��                                    By:�4  �          A&�\������ÿ+��l(�CT!H������\?�\)A
=CS0�                                    By:��  �          A$  �33���;Ǯ��CS�{�33��33?���A#�
CRaH                                    By:�  �          A$Q��
{��녾k���ffCTǮ�
{��@�A6{CR�R                                    By:�&  �          A   �Q���(��#�
�aG�CV:��Q����@�RAN{CS�q                                    By;�  �          A!G��
=��(�>�\)?���CW���
=����@(Q�Aqp�CTǮ                                    By;r  �          A!G��{��=q�#�
�c�
CU���{��p�@ffA@��CS��                                    By;)  �          A�33���Y������CS�33���?��H@�
=CR�H                                    By;7�  �          A�����������  ��Q�CS��������?�{@�(�CS��                                    By;Fd  �          A
=��\)���R�k����CS�{��\)���?�33@�  CS�{                                    By;U
  �          A33������33�n{���CT�
��������?�Q�@�CT�\                                    By;c�  �          A  � z����׿0����p�CT  � z���(�?���A��CS@                                     By;rV  �          A������G��#�
�vffCT=q������
?�Q�A�CS^�                                    By;��  �          A�����\��p��L����=qCV�����\��G�?�
=A	�CV�                                    By;��  �          A�������zῂ�\��=qCVJ=�������H?�(�@陚CV{                                    By;�H  �          A����\������\�\CW
=���\��{?�G�@��CV�=                                    By;��  �          A{��\)���ÿ��
���CUu���\)��  ?�z�@�p�CUT{                                    By;��  �          AG���\��
=�
=�b�\CSaH��\����?�Q�A
�RCRu�                                    By;�:  �          A��� z���=q�\(���CTE� z����?�p�@�p�CS�H                                    By;��  �          A\)��=q��p���  �ÅCUxR��=q��z�?��@�{CUT{                                    By;�  �          A\)� Q���\)���H��CS�
� Q�����?�G�A�RCR�                                    By;�,  �          A33�����;��1G�CS=q����?��
A��CR{                                    By<�  �          A\)��R����<#�
=uCRE��R���
?�33A9CP(�                                    By<x  �          A{��R��=q���
���HCQ:���R��=q?\Ap�CO�H                                    By<"  �          A��\���H���
��\CQY���\���?޸RA+�COz�                                    By<0�  �          A�R�����
=���L��CPY�������
?޸RA*�\CNn                                    By<?j  �          Az����=q��Q���CP�����\)?�(�A&�HCNٚ                                    By<N  �          Ap�����ff>\@��CO����|(�@
=AK33CL��                                    By<\�  T          A����R��\)>aG�?�ffCP���R����?�(�A>�RCM�
                                    By<k\  �          Az��=q��Q�=��
>�ffCPG��=q���
?���A3\)CN#�                                    By<z  �          A33�z������\)�^�RCP�z���\)?�z�A"{CO�                                    By<��  �          A������R�#�
���COǮ�����
?�(�A%p�CM�                                    By<�N  �          A����
��(���\)��(�COG���
����?�z�A   CM��                                    By<��  �          A������=q>k�?���CMp�����i��?�A0Q�CK�                                    By<��  �          Ap��33�~�R>�=q?�z�CM:��33�c33?�A3
=CJ�\                                    By<�@  �          A{�  �}p�>�Q�@�CL�R�  �`  ?��A9G�CJaH                                    By<��  �          Ap����z=q>k�?�CL�=���`��?�  A,��CJ}q                                    By<��  �          A�R��\�g���G��0  CKG���\�`  ?�G�@�33CJ�)                                    By<�2  �          A��(��c33��\)��  CJ���(��XQ�?���@�G�CI��                                    By<��  �          A��	�e<��
>\)CJ���	�R�\?�Az�CH�                                    By=~  �          AG��	���g�=�G�?+�CJ�
�	���R�\?\ACH�                                    By=$  �          Az��Q��j�H>B�\?�
=CKO\�Q��S33?�{A�
CI33                                    By=)�  �          A���H�mp�>��@<(�CK� ��H�N�R?��A<z�CH�R                                    By=8p  �          A33�(��]p�>��H@B�\CJ��(��?\)?�ffA4(�CGc�                                    By=G  T          A\)�z��Z=q?��@n�RCIǮ�z��:=q?��A<z�CF�\                                    By=U�  �          A�
����Z�H>�
=@(��CI� ����>�R?�(�A+�
CG33                                    By=db  �          A��	��S33?G�@��\CI��	��/\)@   AHQ�CE�q                                    By=s  �          A�H�	�C�
?L��@���CG�
�	� ��?�
=ABffCDB�                                    By=��  �          A�����4z�?z�H@ƸRCF:�����{@   AK�CB��                                    By=�T  �          A{�	���9��?��
@�ffCF�)�	���G�@z�AQG�CB��                                    By=��  �          A���	��9��?�G�@���CF���	���\@�
AQG�CB�                                    By=��  �          A�����5?��\@�
=CFff�����R@�\AP(�CB�)                                    By=�F  �          AG��	G��5?k�@�=qCFQ��	G����?���AE�CB�=                                    By=��  �          A���	��1G�?p��@�CE�=�	��(�?�
=ADQ�CB=q                                    By=ْ  �          Ap�����7�?��@�z�CF�������R@
=AW
=CB�
                                    By=�8  �          A����
�=p�?�33@陚CG:���
��\@��Aa�CC�                                    By=��  �          AQ���
�8Q�?xQ�@�ffCF����
��@ ��AMCC�                                    By>�  T          A���	���*=q?n{@�p�CE+��	���?��A@(�CA��                                    By>*  �          A  �  �4z�?xQ�@��CFaH�  �\)?�(�AJ�RCB�q                                    By>"�  �          A���
�.�R?��@��CE�
��
��@ ��AN�HCB�                                    By>1v  �          AQ�����(Q�?��
@ҏ\CE{�����\?��HAH��CA^�                                    By>@  T          A���	���*=q?�  @��
CE(��	���z�?�Q�AF=qCA�=                                    By>N�  �          A��	�(��?���@�Q�CE��	��?��RAK33CAB�                                    By>]h  �          Az��	��)��?��@أ�CE(��	���\?��RAL  CAc�                                    By>l  �          Az��	p��!G�?�\)@�(�CD\)�	p���z�@   ALz�C@�                                     By>z�  �          A���	�\)?�  @��RCD)�	����@ffAW\)C?�3                                    By>�Z  T          A���	G��!�?�z�A�HCDn�	G���@��Ag\)C?��                                    By>�   �          Az��	���(�?�p�@��HCC���	�����@�
AS�C?�q                                    By>��  �          A���
�\�{?�A��CB\)�
�\��G�@
=qA\��C=�f                                    By>�L  �          Az��
�\�Q�?���A
=qCA���
�\���H@�
AS33C=�\                                    By>��  �          A  �
=��{?��A�\C@\�
=��Q�?��RAL��C;�\                                    By>Ҙ  �          A���	�(Q�?=p�@�\)CD�R�	�	��?�
=A+�CB�                                    By>�>  �          A=q�p��j�H�L�;���CL\)�p��Z=q?��A
=qCJ޸                                    By>��  �          Ap�� Q��mp�=�G�?5CL��� Q��X��?�  A(�CJ�                                    By>��  �          Ap����`  <�>k�CKY����N�R?���A��CI�R                                    By?0  �          A  �p��#�
>�z�?�CE��p����?�p�ACC#�                                    By?�  �          A33�	�����\���W
=C<n�	����Q�>�ff@=p�C;�f                                    By?*|  �          A��ff����z��CC�=�ff��
?�R@�=qCCc�                                    By?9"  �          A33�  �-p���  �У�CF0��  �'
=?J=q@�ffCE��                                    By?G�  �          A33��
=��>��
@Q�CR����
=�\)?��HAPQ�CP�                                    By?Vn  �          A
=��z���G�>�  ?�
=CS���z����?�
=AM��CQ{                                    By?e  �          A
�\���}p�>�\)?�=qCOG����dz�?�p�A9�CL�                                    By?s�  �          A
�\��Q��u>8Q�?�(�CNY���Q��`  ?���A*�RCLE                                    By?�`  �          A	��=q����?+�@�{CP��=q�_\)@�Ad  CL�                                    By?�  �          A�����\)?xQ�@У�CP
=���U@Q�A���CL�                                    By?��  �          A	���Q��c33?8Q�@��\CL�{��Q��A�?��RAV�RCIQ�                                    By?�R  �          A  � Q��XQ�?J=q@��CJ�
� Q��6ff?�p�AQ��CG�=                                    By?��  �          Az���R�\?�@o\)CJ\��5?�  A7�
CGO\                                    By?˞  �          A��ff�E�>�(�@5CH��ff�,��?��A"�\CFT{                                    By?�D  �          AQ�� z��\��?#�
@�p�CK=q� z��>{?�{AD��CHL�                                    By?��  �          A���g�?�@\��CL.���J�H?�=qA>�RCIs3                                    By?��  �          A{��\)�s33?
=@u�CM}q��\)�Tz�?���AK�CJ�{                                    By@6  �          A
=� Q��x��?\)@g
=CM�H� Q��Z=q?���AJ{CK�                                    By@�  �          A�����u�?�@U�CMG�����W�?��ABffCJ��                                    By@#�  �          A������qG�>���@$z�CMG�����W�?�  A6�RCJ�
                                    By@2(  �          @���ȣ��J�H=�G�?\(�CN���ȣ��:=q?�G�A$��CL��                                    By@@�  �          @���z��I��>�?}p�CM^���z��8��?��\A
=CKz�                                    By@Ot  �          @�  ��Q��'
=?@  @��\CG���Q��	��?�33AE�CD��                                    By@^  �          @�  ��ff�2�\?(��@��CI0���ff�
=?У�AB�RCF!H                                    By@l�  �          @��
���
�HQ�?W
=@У�CMJ=���
�'
=?�AqG�CI��                                    By@{f  �          @�(���=q�QG�?8Q�@��CN� ��=q�2�\?�{Ai�CK                                      By@�  �          @�=q�љ��J=q?J=q@�p�CM���љ��*=q?��An{CJ{                                    By@��  �          @�=q��\)�0��?8Q�@��CJO\��\)��
?�AS�CF��                                    By@�X  �          @�������:�H?Tz�@�Q�CK��������H?�=qAg�
CH                                    By@��  �          @����Ӆ�9��?n{@�(�CK��Ӆ�
=?�Atz�CG��                                    By@Ĥ  �          @�ff���H�1G�?Y��@�Q�CJ�=���H��?��Af=qCG\                                    By@�J  �          @�ff���H�0��?J=q@ȣ�CJ�3���H��\?�p�A^=qCG(�                                    By@��  T          @��
���
�:=q?^�R@�z�CK�����
���?�{Al��CG�                                    By@�  �          @�=q��33�%���
=�K�CG�R��33�#�
?�@���CG��                                    By@�<  �          @�=q��p��AG�=�G�?^�RCK����p��1�?�Q�A��CI޸                                    ByA�  �          A (���  �Tz�>��
@�CL�H��  �?\)?��RA+�CJh�                                    ByA�  �          @���Å�\(������m�CQ^��Å�w��   �\)CTaH                                    ByA+.  �          @޸R��z��K��L�;\COk���z��?\)?��A��CM��                                    ByA9�  �          @����ʏ\�@  ?
=@���CMaH�ʏ\�&ff?�\)AT��CJO\                                    ByAHz  �          @���{�B�\?�@�33CN&f��{�)��?���AR{CK5�                                    ByAW   �          @�  ��p����H�����4Q�C[  ��p����>�=q@C\L�                                    ByAe�  �          A
=���\����$z���{Cb�����\��ff�Ǯ�0  CeO\                                    ByAtl  �          Ap���z���z������Ca��z���zᾏ\)���Cck�                                    ByA�  �          A���Q����H����C�
C\xR��Q���(�>aG�?�(�C]��                                    ByA��  �          A	���(��������0��CY���(����>�\)?�33C[=q                                    ByA�^  
�          Az��Ӆ��
=��b�RCX���Ӆ��z�L�Ϳ�\)C[33                                    ByA�  �          A���z���{����\CW:���z����R�
=q�k�CZ!H                                    ByA��  �          A����ff��Q��ff�B�\CX�{��ff���\=�\)>�CZ��                                    ByA�P  �          A	p���\)���� ���X(�C]�R��\)���=�\)>�ffC_xR                                    ByA��  �          A	����H�����G��t��C^ٚ���H�\������
C`�q                                    ByA�  �          A	����Q������-p����HC^����Q����
�(����Ca�=                                    ByA�B  �          Az���������H����z�Cas3�����z�xQ���Q�Cd��                                    ByB�  �          A	���������N{���Cc�\�������u����Cg.                                    ByB�  �          A����G����R�E���Cdk���G���p��Tz���=qCg��                                    ByB$4  �          A����(�����e����Cd���(���{��{�=qCh�                                     ByB2�  �          AQ���z�����Z�H��\)Cd��z�����
=��{Chn                                    ByBA�  T          A����{����p��v�\C`�
��{�\���Q�Cb޸                                    ByBP&  �          Ap�������  ��33�S�Cd���������>�z�@   CeY�                                    ByB^�  �          A���R��p������p�Cb
=���R����?E�@���Cb�=                                    ByBmr  �          A��������ff�5����Cc0������ʏ\�5��p�Cf+�                                    ByB|  �          A�R��=q�����Y������Ce����=q��\)��G��p�CiW
                                    ByB��  �          A�������z��\(���Ce�������\)��ff�p�Ch��                                    ByB�d  �          A
=�����{�J�H��(�Cb������ff��\)��33Ce��                                    ByB�
  �          A\)��33���H�1G���p�Cb=q��33��ff�0�����HCe0�                                    ByB��  �          @��R��33�����L����z�Cd����33�������H���Ch�                                     ByB�V  �          @��H�����33�.{���Cd�3�����ff�=p���=qCg��                                    ByB��  �          @���\)��
=�Vff����Cd  ��\)��녿�(��4��Chff                                    ByB�  �          @����������7
=����Ca������H���\���HCd��                                    ByB�H  �          @������H��  �Vff���
C_����H�����ff�9�Cc��                                    ByB��  �          @���Q�����+���ffCa�{��Q���
=�L����Q�CdǮ                                    ByC�  �          @�p�����(�����(�C[0�����(��z����
C^�                                    ByC:  �          @�
=�������>�R���RC_ٚ������
��z��
�RCc�                                     ByC+�  �          @�����H��  �vff��\Cc}q���H��Q�����g\)Ch�H                                    ByC:�  �          @��
��Q���  ��=q��p�Cc�3��Q��\�
=q�\)Cik�                                    ByCI,  �          @���������G���p��33Cbu�������������Chff                                    ByCW�  �          @�������������  ����Ccff������33�
�H��ffCi�                                    ByCfx  �          @�����R�������	z�Ca�f���R��  �%����Ch:�                                    ByCu  �          @�(��{�����(����Cj��{��Ϯ��v�\Cn�=                                    ByC��  �          @��o\)��p��|�����Ckh��o\)��p���
=�i�Co�\                                    ByC�j  �          @�G��o\)��p���z�� �\CkaH�o\)�Ϯ�
=�{�
Cp�                                    ByC�  �          @��
�|(����������G�CiaH�|(�����
��=qCnz�                                    ByC��  
�          @��H�������\��(���CeY���������
���Cj�q                                    ByC�\  
�          @����\)��p������Q�Cf��\)��������{Cl��                                    ByC�  �          @�=q�u���33���H��Cg��u���Q��   ���RCms3                                    ByCۨ  �          @��U���Q������	��Cm���U����
�z�����Crc�                                    ByC�N  �          @�G��XQ���33���H�\)Ck)�XQ��\�0  ����Cp�                                    ByC��  �          @�{�L����������-Ck��L����G��`  ����Cr)                                    ByD�  �          @��
�n�R������\)�CgO\�n�R��{�=p����HCm�f                                    ByD@  �          @�G��y��������
��Cg�=�y����z��1����Cm�)                                    ByD$�  �          @�  ������33�Q���p�Ce�������녾�G��N�RCgǮ                                    ByD3�  �          @�����
�����W��˙�C_�����
��
=�����<��Cd33                                    ByDB2  �          @�=q��\)��Q��Q����Cc����\)��\)����\(�Cf:�                                    ByDP�  �          @�33����ff�+���Cf������Q�333��33Ci:�                                    ByD_~  T          A ���������\�P����33Ces3�����ʏ\����\)Ch��                                    ByDn$  T          A z�����������{ChE����{�
=q�z�HCl�q                                    ByD|�  T          @�����G���(���G���
Ca�{��G���Q��%�����Cg�                                    ByD�p  �          @������������R�Q�C]�
������z��'���=qCd��                                    ByD�  �          @�����ff��G��fff��Ca����ff��{����e�Cfp�                                    ByD��  �          @������\��z��|(���
=C\�\���\��p������Cb�\                                    ByD�b  �          @������\��G��c�
�ڏ\CW����\���R��z=qC]�f                                    ByD�  �          @��R��33�����G
=����Ca�R��33������\�33Ce�{                                    ByDԮ  �          @����Q���\)�'�����C`����Q���G��Tz����Cc��                                    ByD�T  �          @�Q���33��  �   ���RC_
��33��G��J=q��(�Ca��                                    ByD��  �          @���������R�   ����C`��������\)�:�H���HCc=q                                    ByE �  �          @��R����Z�H�AG�����CQ\�����ff����XQ�CVG�                                    ByEF  T          @�ff������=q�����(�CX�\�������
�xQ���\C\(�                                    ByE�  �          @�\)������\)�(����CY�\�������׿h���ٙ�C\�q                                    ByE,�  �          @������{�7���\)CZ������
�����RC]��                                    ByE;8  �          @�=q�����(��0  ����CX�q������׿�  ��C\��                                    ByEI�  �          @���������H����(�C_�������zῺ�H�,Q�Cc��                                    ByEX�            @������H��
=�E����C]� ���H��ff��(��/
=Ca�
                                    ByEg*  
�          @�G�������H�P�����C_5�������
�����=Ccp�                                    ByEu�  �          @���������p��Tz���{C`aH������ff����B�\Cd�)                                    ByE�v  
�          @�Q����R��ff�S�
���
Cc�{���R���R����7�Cg��                                    ByE�  
�          @�=q��������[���  Ch+�������H�Ǯ�7�Ck�                                    ByE��  �          @����R��{�w����Ch@ ���R�˅�G��mG�Cl��                                    ByE�h  
�          @�(���=q��
=�U����Cf+���=q��\)��  �0Q�CiǮ                                    ByE�  �          @�=q��p���Q��\����p�Ca���p����\��G��O�
Ce�\                                    ByEʹ  �          A����p����E���RC[Ǯ��p���(���Q��   C_�{                                    ByE�Z  
�          A�����p��<�����C[�������\�����
=C_J=                                    ByE�   �          A����������A���p�C]z�������G�������Ca
=                                    ByE��  �          A �����
���
�>�R��z�Ce����
��Q쿑���\Ch!H                                    ByFL  
�          @����(��Å�/\)���Ck����(����ͿE���(�Cn#�                                    ByF�  
�          @��
�������1���Q�Cf������Ǯ�s33��
=CiB�                                    ByF%�  �          @�\)��(���
=��H��(�Cd=q��(���{�+����RCf�
                                    ByF4>  �          @�(���=q�����\)���Ccp���=q��{���H�c33Ce�                                    ByFB�  T          Aff��{��{�`���˙�C[xR��{���׿����\��C`!H                                    ByFQ�  
�          @�����ff�a��g����CR����ff��ff�=q����CX�                                    ByF`0  �          @��H�����{�z�H����CZB��������   ��G�C`33                                    ByFn�  �          @�����H��z���=q�ffCgu����H��{�'
=����Cl�                                    ByF}|  �          @����Q�����aG���p�C\}q��Q����\���rffCa^�                                    ByF�"  	�          @����������"�\��
=C]
�������H��G���
=C`�                                    ByF��  �          @�Q���(���
=�9����=qC]J=��(����
�����$z�C`�                                    ByF�n  �          @�
=��(���  �7
=��(�C`@ ��(����
���\�\)Cc��                                    ByF�  �          @�{��=q���
�$z���
=C^� ��=q������
��  Ca�                                     ByFƺ  T          @�Q������
�&ff��G�C]�����p�������C`�R                                    ByF�`  
(          @��\��(����H��{�[�
C^!H��(������z����C_�H                                    ByF�  "          @�������������"�HC]�������>\)?�=qC^\                                    ByF�  T          @�z����
��ff��\)�%�C]aH���
��(�=�?n{C^k�                                    ByGR  
�          A{��=q��ff��z�� z�Ca�f��=q���>���@�Cbu�                                    ByG�  
.          AG���(����
�k���\)Co��(�����?�\AK�Cn(�                                    ByG�  
�          @���xQ��ƸR@5�A�
=Cn�xQ���z�@�  B��Ch�                                    ByG-D  �          A
�R��������Q�
=Cd�������?�  A>=qCc��                                    ByG;�  �          A=q�ƸR��{���=p�Cc��ƸR��ff?�ffA6=qCb�                                    ByGJ�  T          A(�������ü��W
=Cc�������  ?�p�A?�Cb�                                     ByGY6  
�          Ap����ۅ>��?^�RCa����G�@AJ{C`\)                                    ByGg�  �          A\)��=q�أ�?+�@�G�Ca����=q�ə�@#�
A{�C_�
                                    ByGv�  T          A����  ��(���Q�
=qC_c���  ����?�(�A+
=C^O\                                    ByG�(  "          A
=����љ��\)�fffCa������ʏ\?��HA,Q�C`��                                    ByG��  �          A�\���R�ȣ��(��c�Cbp����R��(���{�
=qCd�                                    ByG�t  �          A33���\���1G���33Cg�����\��ff�Y�����
Ci�
                                    ByG�  
�          A\)����  ���C�Ch������  =�?O\)Ci}q                                    ByG��  T          A�
�����=q���
�ڏ\CkǮ�����\?}p�@��
Ck��                                    ByG�f  
Z          AG����R�����\�(�Ckff���R��  ?J=q@��
Ck�f                                    ByG�  �          A�������녿�p���Ch�����޸R>��@ECh�f                                    ByG�  
�          A=q�������$z����\Ce0�������z�.{��p�Cg!H                                    ByG�X  
�          A(�����녿���F�\CfaH���ڏ\<#�
=�Q�Cgz�                                    ByH�  "          A
{������z�\�%�Cb.������Q�?�=qA33Ca��                                    ByH�  T          AG����������
�+�C`}q������(�=�Q�?(�Ca}q                                    ByH&J  T          A(���\)���H��z��Q�Cg�{��\)��p�?��A(  Cf�{                                    ByH4�  �          A	���z��θR��
=�4Q�Cf0���z���>#�
?��Cg�                                    ByHC�  T          A
�H��{������
�Z�RCa�=��{��zᾨ���(�Cc{                                    ByHR<  
�          A	���θR���\��\)�I�C[���θR��(���33�z�C]�                                    ByH`�  �          A\)��(����
�!����HCe�
��(���=q�333���
Cg�q                                    ByHo�  �          A  ��(����H�9������Ci}q��(���(���  ��(�Ck�)                                    ByH~.  "          A  �Å���H�������C](��Å��G��^�R���
C_u�                                    ByH��  "          A���(�������G��
{C[����(���{>W
=?��HC\s3                                    ByH�z  
�          A�R��\)����{�u�CX޸��\)���׿Tz���{C[(�                                    ByH�   �          A���=q��ff�=q��p�CZ���=q��p��}p��ٙ�C\�)                                    ByH��  	�          A(���33��Q���R��p�CWff��33��Q쿘Q���CZG�                                    ByH�l  "          A������p��{�v�HC^s3����=q�0������C`s3                                    ByH�  T          A�������
=��33�7�C^\)������R�\)�p��C_��                                    ByH�  �          A(�������(��9������C]�H�������R��Q�� ��C`��                                    ByH�^  �          A����33��  ����Q��CZ@ ��33��=q���H�Z=qC[��                                    ByI  �          Ap��ҏ\��p�����p�CX� �ҏ\����L�;��
CY�\                                    ByI�  �          A�R�ȣ�����z��k
=CY� �ȣ���{�=p���p�C[��                                    ByIP  �          A33����(��
=�n�HCW�����׿Tz���G�CY�q                                    ByI-�  
�          @�����H���\�p���ffCVL����H��  ��  ��
=CXٚ                                    ByI<�  
�          @�p���
=�|������{CSaH��
=�����
=���CV=q                                    ByIKB  T          A ����p��N{�   ���CL���p��p�׿�=q�5��CP}q                                    ByIY�  
�          @��R�ٙ��S�
��H��CM���ٙ��u���p��+�CQY�                                    ByIh�  T          @����  �S33�����RCN���  �q녿�����CQ:�                                    ByIw4  
�          @�\)��Q��E��{��p�CK�R��Q��c33��{���CN�
                                    ByI��  
�          A����  �S�
��
���HCMJ=��  �r�\�����CPk�                                    ByI��  
Z          A�\���
�C33�%���33CK+����
�fff���H�ACNٚ                                    ByI�&  
�          A ����ff�8���4z����\CJ�=��ff�`���   �e�CN�=                                    ByI��  
�          A���߮�A��0����G�CKk��߮�hQ��33�X��COp�                                    ByI�r  
Z          A�\��{�'
=�2�\��(�CG����{�N�R��\�g�CL&f                                    ByI�  
�          A�\����G��7
=��ffCEQ�����:=q�(��y�CI�
                                    ByIݾ  
Z          A{���
=�/\)��G�CC�����.�R���qp�CHW
                                    ByI�d  	�          A�H������\�'���\)CE:������8Q���H�]CI@                                     ByI�
  �          AQ���{�{�)�����CFaH��{�C33�����Z{CJT{                                    ByJ	�  
Z          A�H���
���(����z�CE�����
�<�Ϳ��H�^=qCI��                                    ByJV  
�          Aff����	���(����33CD0�����/\)�G��e�CHJ=                                    ByJ&�  
(          A���  �\)�!���
=CD����  �333����T��CHn                                    ByJ5�  f          A����\��H�Q���\)CE���\�;����H�>=qCI!H                                    ByJDH  �          A  ���=q������CE����8�ÿ�\)�5G�CH��                                    ByJR�  �          A����Q����'
=��ffCE���Q��<(���Q��X��CI\)                                    ByJa�  �          Ap����)���*�H��CG�����N{��Q��W�
CKs3                                    ByJp:  T          A
{��  �����Q��c33CX����  ��(��W
=����CZ�\                                    ByJ~�  �          A
=q�������R�(��j�HCVO\������33�z�H�ϮCXxR                                    ByJ��  �          A	��ᙚ��p��   ���CR���ᙚ�����z��  CUu�                                    ByJ�,  �          A	�����j=q��}CNk������
�������CQ33                                    ByJ��  �          A����b�\��]��CL�q���|(�����
=CO&f                                    ByJ�x  �          A
�H���
�`������j�RCL�����
�|(�������COO\                                    ByJ�  �          A
=� z��'
=���R�Tz�CE�q� z��@�׿���
=CH��                                    ByJ��  
�          A
�\��p��3�
�   �V=qCG����p��L�Ϳ��\�\)CJ�                                    ByJ�j  �          A
ff���R�!��	���g\)CE�����R�>{���R�
=CHxR                                    ByJ�  �          A�
���R�:�H�z��[
=CH!H���R�U�����
�\CJ��                                    ByK�  �          A�� ��� ���  �n�RCET{� ���>{�˅�'�
CHB�                                    ByK\  �          A
=q���*=q�
=q�g
=CF�����E��(��Q�CIL�                                    ByK   �          A
=q�����+��p��m��CF�R�����G��\�"ffCI�\                                    ByK.�  �          A
=��\)�+��
�H�g�CF�=��\)�G
=��p��p�CIL�                                    ByK=N  �          A����R�5�����c�CG�����R�P  ����CJ=q                                    ByKK�  �          A��� (��A녿��H�N{CH�q� (��Z=q�������HCK
=                                    ByKZ�  �          AQ��=q�/\)���
�:�HCF���=q�E������z�CH�3                                    ByKi@  �          AQ�� ���AG���z��/33CH�� ���Tz�k�����CJc�                                    ByKw�  �          A33���\�]p��Ǯ�%�CK�)���\�n�R�8Q���\)CMxR                                    ByK��  �          A(������n�R������CM�������|(����?\)CNǮ                                    ByK�2  
�          A����  �w
=����+\)CNxR��  ��(��5���
CP\                                    ByK��  T          A����=q�n{��=q�%p�CMu���=q�\)�0����\)CO                                      ByK�~  �          Ap�� ���P  ��z��,��CJ  � ���b�\�^�R��p�CK                                    ByK�$  �          A���H�;����
�9CG����H�P  �����޸RCI�3                                    ByK��  �          A����8Q���\�U��CG����QG�����33CI�3                                    ByK�p  �          A�R� ���<���=q�{\)CH�� ���Z�H��
=�.ffCK                                      ByK�  �          A{�ff�(��   ����CD���ff�<(�����D��CG��                                    ByK��  �          Aff����'��%��G�CE�����HQ��Q��I�CI&f                                    ByL
b  �          AQ��ff�3�
�,(����\CG��ff�U���R�L  CJO\                                    ByL  �          A����(��`  �-p���\)CK�3��(����׿�{�=�CN�q                                    ByL'�  �          A�����u��0  ����CN33�������H���:ffCQ.                                    ByL6T  �          A{��=q�w
=�*�H���CNE��=q��33��  �0��CQ�                                    ByLD�  �          A�H���\�y���1G���z�CNz����\��p����9�CQh�                                    ByLS�  �          A{��=q�w��*=q���CNT{��=q�����  �0��CQ!H                                    ByLbF  �          A33���\����&ff��\)COp����\���ÿ�33�%�CR{                                    ByLp�  �          A����p  � ���
=CMJ=�����R�У��$z�CO�                                    ByL�  �          A(�����XQ��=q�x��CJ������tz��\)�&{CM�\                                    ByL�8  �          A33�   �z=q�33�g�CN
=�   ��녿�33��
CPT{                                    ByL��  �          A33� (��o\)�!��
=CM�� (���ff��z��&=qCO�f                                    ByL��  �          A
=� Q��e��*�H��p�CL
=� Q���=q���9�CN�f                                    ByL�*  �          A�����xQ��!G���ffCN@ ������\�У��$��CP�{                                    ByL��  �          Az���
=��(��(Q����CQ�{��
=���H�У��"ffCT�                                    ByL�v  �          A����33���ÿ����B{CR���33��33�p�����CS�                                    ByL�  �          A���ff���Ϳ�Q��A�CP����ff���R�xQ���
=CR��                                    ByL��  �          A�
����p���33�  CO+������
���H�C�
CPJ=                                    ByMh  �          A(�� z���Q�c�
����CQT{� z���33=��
?   CQ�=                                    ByM  �          A\)���R����z�H��z�CT� ���R��Q�=�\)>�(�CU�                                    ByM �  �          A���=q��Q�+���\)CU���=q��G�>Ǯ@{CU�                                    ByM/Z  �          A(���
=�����Q���
CX�{��
=��=q?B�\@��CX\)                                    ByM>   �          A33��z����H�&ff��ffCX����z����>��@@��CX�f                                    ByML�  �          A=q���H��Q�p����=qCX�����H���H>.{?���CX�R                                    ByM[L  �          A�
��\)��{��z���p�C]G���\)���=#�
>�  C]�)                                    ByMi�  �          A��أ�������(��\)CZ\�أ������33�33C[
=                                    ByMx�  �          A�������R������C[Q������(��u��ffC\+�                                    ByM�>  �          A
�R������p�����>�\CW���������=p���CX�                                    ByM��  �          A���
=��{�޸R�7�
CWT{��
=��{�0����=qCX��                                    ByM��  �          A
ff��G���p����DQ�CU����G���ff�Y������CW�                                    ByM�0  �          A������׿�\�;\)CS�R����G��Q���z�CUs3                                    ByM��  �          AQ���  ��z��{�D(�CP�H��  ���}p��ϮCR��                                    ByM�|  �          A  ��=q���
�����N{CR����=q��p����
��G�CT��                                    ByM�"  �          A������\��\�YG�CT�=����zΉ���CV��                                    ByM��  �          A
{������(Q���Q�CR��������
��  �;\)CU0�                                    ByM�n  �          A���\��33�@  ��33CR���\����Q��b=qCU\                                    ByN  �          A\)��  ���R�����z�CT� ��  ���>�z�@ ��CT�                                    ByN�  �          A33�ڏ\����?�G�A	CW\�ڏ\��ff@�Ao�
CU
                                    ByN(`  �          A�
��ff��  ?��
A&{CX� ��ff���@�RA��
CV�                                     ByN7  �          AQ���  ��Q�@�RAs�CY����  ��\)@J�HA��CV��                                    ByNE�  �          A���������?���AP(�CXٚ�������H@0  A�ffCV+�                                    ByNTR  �          AG���=q��(�@��Axz�CV� ��=q�x��@<��A�p�CS�{                                    ByNb�  �          A33��(����
?�ffA.{CV�f��(���\)@�HA��RCT�{                                    ByNq�  �          A�\��(�����?��Az�CVs3��(���ff@\)A~�\CTT{                                    ByN�D  �          A{��33��33?�A[
=CUk���33�y��@.{A��\CR�)                                    ByN��  �          A
=�������?�ffA.{CWQ��������@=qA��RCU
=                                    ByN��  �          A���G�����?�z�AG�CXJ=��G���{@33A��CV5�                                    ByN�6  �          A  ��Q���
=?���A/\)CUu���Q����H@��A���CS(�                                    ByN��  �          AQ���33���?���AO\)CS���33�s�
@'�A�  CQ�                                    ByNɂ  �          A����{���?�=qAK�
CR����{�o\)@%�A�
=CPW
                                    ByN�(  
(          A\)���
���H?���AL��CR����
�k�@#33A�
=CP&f                                    ByN��  �          A�\�أ����@33Ai�CR��أ��fff@1�A��CP                                    ByN�t  �          A�
��\)�|��?�33AUG�CQ����\)�a�@&ffA�\)CN�{                                    ByO  �          A��������?��HA>�HCR\����j�H@(�A��HCO��                                    ByO�  �          A(��޸R��?��A,(�CR�q�޸R�tz�@�\A��RCP�                                    ByO!f  �          AG���Q���p�?�A9G�CRǮ��Q��r�\@=qA��RCPn                                    ByO0  �          Ap���\)�\)?���@�33CP�f��\)�n{?���AI��CO:�                                    ByO>�  �          A  ��=q��z�?fff@ǮCRY���=q�y��?��A7�CP��                                    ByOMX  �          A  ��(����?�ffA��CTW
��(�����@z�Ahz�CRu�                                    ByO[�  �          A�
�ٙ���ff?�ffA�CU(��ٙ���z�@�Aj�RCSL�                                    ByOj�  �          A(��љ���  ?�33A��CW���љ���p�@\)A}p�CV�                                    ByOyJ  �          A���(���33?�Q�A[
=C[��(����@5�A�p�CYE                                    ByO��  T          AG���=q����?��\A��C[�\��=q��ff@��Au�CZ{                                    ByO��  �          A{�������?��
A��C\�R�������H@�RAv�HC[                                    ByO�<  �          A��������33?�33@�ffC\u��������@AhQ�CZ��                                    ByO��  �          A�H��
=��=q?xQ�@�(�C[p���
=���?�33AP��CZ�                                    ByO  �          A
=�ҏ\��(�?�p�AffCY�3�ҏ\���\@�Ai�CXL�                                    ByO�.  T          A�
��33��p�?��AG�CZ\��33��33@�RAs�
CXO\                                    ByO��  �          A��˅��?���AO
=CY�=�˅����@,(�A��CWu�                                    ByO�z  �          A(���(����?��A2ffC^����(�����@'
=A���C\�q                                    ByO�   �          A������{?˅A-C_.�����=q@#�
A���C]J=                                    ByP�  �          A�������p�?��
A
�RCc���������@�A�Cb�                                    ByPl  �          A�\��ff��G�?��AG�Cb����ff��\)@z�A�  Ca!H                                    ByP)  �          A����{���?�{@�Cd�
��{��(�@	��Ap��Cc@                                     ByP7�  �          A=q��33��  ?��@�33CfǮ��33��
=@ffAo\)Ce�                                    ByPF^  �          A���{��
=?��@��Cj\��{��ff@Q�AuG�Ch޸                                    ByPU  �          A �����У�?\(�@��HCjW
������?���A`Q�CiO\                                    ByPc�  �          @�ff��ff��33?Tz�@�Q�Ciu���ff�Å?�33A]�Chn                                    ByPrP  �          A z���(����H?
=@�
=Chh���(�����?�z�A>�\Cg�\                                    ByP��  �          A�
��ff��?��A33Chh���ff���
@Q�A��\Cg                                    ByP��  �          A����ff��p�?:�H@��Cg  ��ff��ff?�ffAHz�Cf
=                                    ByP�B  �          A����33��G�?8Q�@�\)Ce����33�\?�G�ADQ�Cd��                                    ByP��  �          A  ��=q��
=?z�H@�=qCe� ��=q��
=@   A`��CdT{                                    ByP��  �          A���Q����H?�Q�A!G�Cg�R��Q�����@{A�(�Cf:�                                    ByP�4  �          A=q��(����H?�Q�A#�Chk���(�����@{A���Cf�3                                    ByP��  �          Ap����
�ə�?���A%�ChL����
��
=@{A�{Cf��                                    ByP�  �          A�����ə�?��\A  Cg������Q�@�A�
=Cf��                                    ByP�&  �          Aff��
=��{?E�@�ffCe޸��
=��\)?�\AICd�f                                    ByQ�  h          A����
��{?�R@��\Ce����
��Q�?�\)A5�Cd5�                                    ByQr  �          A(���
=���?(�@��RCdn��
=��\)?���A2�\Cc�{                                    ByQ"  �          A(����H���?���@�p�Cg�=���H����@�AnffCf\)                                    ByQ0�  �          A\)��33�ʏ\?�
=A��Cg&f��33����@(�Aw33Ce��                                    ByQ?d  T          Aff���R��ff?c�
@�Q�Cf  ���R��\)?�\)ATz�Cd�R                                    ByQN
  T          A\)���R��z�?�G�A{Ch+����R�Å@G�A��RCf�f                                    ByQ\�  �          A����\��Q�?�{A7�ChT{���\��p�@%�A�\)Cf�=                                    ByQkV  �          @�p������z�?�
=ADQ�Ck�R�����G�@*�HA�  Cj5�                                    ByQy�  �          @�{���
�У�?�\)AY��Cm�q���
��z�@7�A��
Cl.                                    ByQ��  �          A Q��dz���ff?޸RAIG�Cr���dz����H@4z�A���Cq�=                                    ByQ�H  �          A ���X���޸R@�RA�=qCt�X������@R�\A���Cr��                                    ByQ��  �          A���L����ff@'�A��CuB��L���θR@j�HA�{Cs��                                    ByQ��  �          A(��AG���@#�
A��\CwW
�AG���  @i��A�Q�Cu��                                    ByQ�:  �          A���Fff���@#33A�\)Cv޸�Fff����@i��AУ�Cuh�                                    ByQ��  �          A���L(���p�@,��A�  Cv
=�L(���p�@q�Aأ�Cts3                                    ByQ��  �          AQ��C�
��R@(��A���Cw�C�
��
=@n{A��
Cu��                                    ByQ�,  �          A ���>{��=q@�HA�{Cw5��>{���
@^�RA���Cu��                                    ByQ��  �          A (��*=q��\?��AN�\Cz��*=q��
=@8��A�33Cy)                                    ByRx  �          A���.�R��
=@�A���CyL��.�R����@\(�A��
Cx�                                    ByR  �          A Q��$z��޸R@C�
A��Cy� �$z����@��\A���Cx0�                                    ByR)�  �          @�
=�"�\��p�@a�A�ffCy+��"�\���@�  B	z�CwJ=                                    ByR8j  �          @����#33��  @k�Aޏ\Cx���#33���
@��
BG�Cv�=                                    ByRG  �          @��P  ����@/\)A�ffCs��P  ��p�@mp�A��Cr.                                    ByRU�  �          A���,����Q�@\)A�  Cw}q�,�����H@��Bz�Cu5�                                    ByRd\  �          @�\)�*=q��G�@mp�A�  Cw�{�*=q���@�z�Bz�Cu�                                     ByRs  �          @���;��θR@R�\A�\)Cu���;�����@��RBz�Cs�)                                    ByR��  �          @�\)���R��(�?8Q�@�
=C\G����R��
=?��A2�\C[O\                                    ByR�N  �          @��H��{�?\)���H�@Q�CM����{�J�H��G��  CO&f                                    ByR��  �          @߮��  � ���33����CE�
��  ����p���z�CHk�                                    ByR��  �          @�
=��p������z����CE����p��G�� ����(�CH+�                                    ByR�@  �          @�ff��
=�HQ쿾�R�G
=CO�3��
=�S�
���\��CQ\                                    ByR��  �          @������b�\�����/�
CS^������l(��J=q�ҏ\CTu�                                    ByRٌ  �          @�����{���������
CZ�f��{����?�@���CZz�                                    ByR�2  �          @�=q������  �B�\��p�CX�H������\)>�ff@\��CX�                                    ByR��  �          @������녿��R�E�CGz����{��33��CH�R                                    ByS~  �          @�{�Ǯ������Q��F�\CET{�Ǯ�Q쿓33�CF޸                                    ByS$  �          @�(���(��.{�ٙ��m��C:���(��k��˅�]G�C<+�                                    ByS"�  �          @Ӆ��z�&ff��=q�\��C9�\��z�^�R��(��Mp�C;��                                    ByS1p  �          @�p���ff�^�R��
=�FffC;�3��ff���ÿ���333C=aH                                    ByS@  �          @��
��녿�
=��\)�>�RC@�q��녿�{��33� z�CBB�                                    BySN�  �          @ҏ\��=q���
�L�Ϳ��HCC����=q���=u>�CC�\                                    ByS]b  �          @�p��ƸR��������$z�CI!H�ƸR��H=#�
>�{CIE                                    BySl  "          @�(����H�%�����b�\CJ�����H�'
=�L�;���CK8R                                    BySz�  �          @ҏ\�Å��;W
=��CI���Å���>�?���CI޸                                    ByS�T  �          @ҏ\���
�=q���
�5CI�=���
���>�  @��CIp�                                    ByS��  �          @�{�Ǯ���>k�?�p�CI{�Ǯ�ff?\)@���CH��                                    ByS��  �          @����G��!�>�=q@�CJ����G��{?(�@��HCJ:�                                    ByS�F  �          @�ff�����1�>��@k�CM�)�����,��?J=q@���CL�3                                    ByS��  �          @����Q��1G�>���@c33CM����Q��,(�?E�@�z�CM                                    BySҒ  �          @�������0  >�p�@U�CMz������+�?=p�@���CL޸                                    ByS�8  �          @�33���H�<��>�(�@y��CO�{���H�7�?Tz�@�\)CO&f                                    ByS��  �          @˅���R�L(�?!G�@��RCRQ����R�E�?��A��CQn                                    ByS��  �          @�(����P��?J=q@��HCR�3���G�?�(�A/�
CQ��                                    ByT*  �          @�p���{�QG�?h��AffCS  ��{�G�?��A@z�CQ��                                    ByT�  �          @�
=����R�\?n{A��CR������HQ�?�{AB�\CQ�3                                    ByT*v  �          @�ff��
=�S33?Tz�@�z�CS���
=�J=q?�G�A4(�CR                                    ByT9  �          @Ϯ��  �L��?�G�A2�HCR+���  �@  ?�An�\CP��                                    ByTG�  �          @�����N�R?
=q@�
=CQ�f���HQ�?uA�\CP��                                    ByTVh  �          @�����n�R�aG�����CW���n�R>�=q@��CW�q                                    ByTe  �          @������R����333���C_
=���R��
=�����C_k�                                    ByTs�  �          @�{�����vff�E�����CZ�{�����z=q����(�C[O\                                    ByT�Z  �          @�\)�����u������'\)CZ�\�����|�Ϳ�R��  C[�H                                    ByT�   �          @�33��G����׿�G��\)C\��G����
���H���C\��                                    ByT��  �          @�\)��
=�����h����C[(���
=��(��\�W�C[��                                    ByT�L  �          @У���(��}p��333��CY�f��(����׾B�\��z�CZ�                                    ByT��  �          @�z������
�\)���\CZ�����������  CZ                                    ByT˘  �          @�Q���ff��ff�����)��C\J=��ff��ff>u@�C\Q�                                    ByT�>  �          @�=q���\��p��#�
���
C[^����\��z�>��H@�G�C[+�                                    ByT��  �          @�
=������R�#�
�uC\�������p�?�@���C\Y�                                    ByT��  �          @�ff���
��\)<��
>B�\C\����
��{?��@�p�C\��                                    ByU0  T          @Ϯ��G����H���ͿY��C[)��G���=q>�
=@l(�CZ�R                                    ByU�  �          @�
=������33=���?\(�Cb0���������?5@���Ca��                                    ByU#|  �          @�p��������>B�\?�33CcaH������33?L��@���Cc                                    ByU2"  �          @����p���G�>�ff@|��C]
��p���ff?z�HA	C\�                                     ByU@�  �          @��H��p���  >�\)@!G�C^Q���p���{?L��@���C]�)                                    ByUOn  �          @����������
�����RC_!H������33>�@���C^�                                    ByU^  �          @�\)��33��33���
�W
=C_E��33��=q>�@��C_�                                    ByUl�  �          @�33�������\=�G�?��C^������G�?(�@�Q�C]�3                                    ByU{`  �          @���������  >�p�@[�C\�������{�?Y��@�\)C\33                                    ByU�  �          @����G��H��?��@�  CS���G��C33?k�AffCS33                                    ByU��  �          @������R��(�?0��@ۅCE� ���R�У�?c�
AffCD�{                                    ByU�R  �          @��\���ÿ�Q�?.{@��CE���ÿ���?aG�A
=qCD�                                    ByU��  �          @�����׿���?n{A
=CCٚ���׿�Q�?�{A0(�CB��                                    ByUĞ  �          @�  ����ٙ�?p��A=qCEn�������?���A6{CD8R                                    ByU�D  �          @����=q�
�H?Q�A=qCJ(���=q�33?���A*�RCI�                                    ByU��  �          @���{��
?   @�=qCJ����{��R?E�@�RCJL�                                    ByU�  �          @�(���ff�z�?8Q�@���CH�=��ff��(�?uAffCG޸                                    ByU�6  �          @��R��p��G�?��\A�CJ���p����?��
AF{CI��                                    ByV�  �          @�������\?z�HAz�CK+�����
=q?��RAB{CI��                                    ByV�  �          @�\)���?fffA
=CKJ=���p�?�A4��CJ0�                                    ByV+(  �          @������G�?E�@�(�CJs3����
=q?��
AffCI�                                     ByV9�  �          @�33���H��R?\(�A�HCJ�����H�
=?�{A/33CI�\                                    ByVHt  �          @��\��G��33?���AVffCI=q��G����?�=qA{�
CG�)                                    ByVW  T          @��\���ÿ��R?���Af�RCH����ÿ�?�A�p�CF�                                    ByVe�  �          @�ff����=q?��RAs33CGz�����33?ٙ�A��\CE��                                    ByVtf  �          @��R��\)��33?���A�CE� ��\)����?�\A���CC�=                                    ByV�  �          @�=q���ÿ�\)@�\A�Q�CB�����ÿ���@(�A�z�C@
=                                    ByV��  �          @��H���R��(�@A��HCA33���R�s33@p�AǙ�C>J=                                    ByV�X  �          @���������ff@
=qA�Q�CBn�������@33A�(�C?�                                    ByV��  �          @�33���׿�ff?��A��\CF�
���׿�=q?�p�A��\CD�R                                    ByV��  �          @�
=���\���?˅Aw�
CI����\����?���A���CH)                                    ByV�J  �          @���=q����@A�G�CDn��=q����@  A��HCA��                                    ByV��  �          @���Q��Q�@ffA���CE�)��Q쿹��@�A�p�CCaH                                    ByV�  �          @�33���H� ��?�  A��CI�=���H��ff?��HA��
CGz�                                    ByV�<  �          @�����{����@�\A���C?���{�Tz�@��A�ffC=�                                    ByW�  �          @�z����Ϳ�=q@G�A��RCE����Ϳ���@�Aģ�CB\)                                    ByW�  �          @��
��  �(��?���AfffCO�
��  �p�?�p�A�ffCN8R                                    ByW$.  �          @�G����
�U�?z�HA��CW�����
�L��?���AW�CV�q                                    ByW2�  �          @�G�����c�
?E�@�{CZk�����\��?�z�A9��CY��                                    ByWAz  �          @�����33�X��?+�@�{CXc���33�S33?��A&�RCW��                                    ByWP   �          @����(��K�?!G�@��CV� ��(��Fff?z�HA33CU�                                     ByW^�  �          @�
=����@  ?@  @��HCS������9��?�=qA.{CS
                                    ByWml  �          @�  ��33�=p�?c�
A33CSff��33�5?��HAB�HCRff                                    ByW|  �          @��R��G��E�    <#�
CSp���G��C�
>�{@QG�CSG�                                    ByW��  �          @��\��z��L(������CU+���z��L(�>W
=@ffCU!H                                    ByW�^  �          @�Q���p��<�;\�p  CR����p��>{�L�Ϳ��CS)                                    ByW�  �          @�{�����E�>aG�@{CT�
�����B�\?��@�=qCT}q                                    ByW��  �          @�33���H�J�H>�\)@:=qCV�����H�HQ�?�R@��CV8R                                    ByW�P  �          @��H�����e=u?�C[�������c�
>�G�@���C[�                                     ByW��  �          @�������XQ쾊=q�.�RCY�����X��=���?z�HCY{                                    ByW�  �          @��H��G��G�?\(�ACL@ ��G��
�H?���A5G�CK@                                     ByW�B  �          @�����p��Q�?uA33CM�
��p��G�?��HAH��CL�q                                    ByW��  �          @�33�����p�?uA{CK�)�����?�Q�ADQ�CJ��                                    ByX�  �          @�=q�����(�?z�HA"=qCK�\������?��HAHQ�CJn                                    ByX4  �          @�=q������?uA=qCN�\����?��HAH��CMz�                                    ByX+�  �          @����(���(�?У�A��CI����(����?�=qA�\)CH�                                    ByX:�  �          @����{��?�
=A���CH!H��{����@
=A��\CE�                                    ByXI&  �          @�����33���
?�
=A�p�CH(���33����@ffA��CE�                                    ByXW�  �          @�Q����Ϳ�{@
=A�=qCCu����Ϳ���@\)A�{C@��                                    ByXfr  T          @����H��ff?��A��CE�R���H���@�\A���CC}q                                    ByXu  �          @�{��(���\)?��HAz{CH����(����H?�33A�p�CGG�                                    ByX��  �          @���(��?�
=AHz�CK(���(�����?��Al��CI��                                    ByX�d  �          @�\)������
?��
AX��CJ�{���Ϳ�?��RA|��CIc�                                    ByX�
  �          @���33��\)?�=qA��CI\��33��Q�?�G�A�p�CGB�                                    ByX��  �          @�(���
=���R@33A��CE�\��
=���\@(�A�33CC�                                    ByX�V  �          @��H���׿�z�@QG�B{CD
=���׿O\)@W�B�C?s3                                    ByX��  �          @�G���p�����@0��A�\)CD����p����
@8Q�B�RCA!H                                    ByXۢ  �          @�����\��ff@(�A��CD�R���\����@A�Q�CB��                                    ByX�H  �          @�����z��  @,(�A�{CE���z῜(�@5�A�\CA��                                    ByX��  �          @�  �����  @#33A��HCE  ������R@,(�A�p�CB)                                    ByY�  �          @�G���ff����@*=qA�ffCD^���ff��
=@2�\A�Q�CA^�                                    ByY:  �          @��\�����Q�@2�\A���CDff�����33@:�HA��HCA=q                                    ByY$�  �          @�\)������(�@1�A�CD�R������
=@:=qA��
CAǮ                                    ByY3�  �          @����(�����@�HA̸RCC
��(�����@"�\A׮C@Q�                                    ByYB,  �          @����
=��  @*�HA�33CB�f��
=�z�H@1�A��
C?�q                                    ByYP�  �          @�p���33�fff@,��A�
=C?���33�!G�@1G�A��\C;�=                                    ByY_x  �          @�33��z�h��@   A���C?)��z�(��@%�A��C<�                                    ByYn  �          @�G���Q�J=q@(Q�A���C=�3��Q��@,��A���C:�H                                    ByY|�  �          @�{��\)��@#�
A��C;J=��\)���R@'
=A�33C8                                    ByY�j  �          @����녿   @!G�A�  C:33��녾u@#33A�C7�                                    ByY�  �          @�����Q�5@(Q�A�33C<����Q��ff@,(�A�
=C9��                                    ByY��  �          @�Q�����z�@
=A��
C;&f�����33@=qA�ffC8B�                                    ByY�\  �          @��R��
=��ff@
=qA��C9c���
=�k�@(�A��C6�=                                    ByY�  �          @�\)������(�@�A��\C9!H�����k�@�
A��C6��                                    ByYԨ  �          @�{��
=�Ǯ@�A��C8����
=�8Q�@	��A��
C6#�                                    ByY�N  
�          @�33�����G�@��A�(�C9}q����k�@
�HAɅC6޸                                    ByY��  
�          @�����녿   @�A׮C:=q��녾�=q@�A�p�C7Y�                                    ByZ �  �          @����׿�R@��A�C;Ǯ���׾\@   A��C8�q                                    ByZ@  �          @�G��������@!G�A���CAaH�����c�
@'�A�{C>�                                     ByZ�  �          @�����(���\)@+�A��CD����(�����@3�
A�=qCAh�                                    ByZ,�  �          @�p���\)��
=@7�A�  CB  ��\)�c�
@>{A���C>��                                    ByZ;2  �          @�
=����Y��@A�B��C>�������@FffBp�C;&f                                    ByZI�  �          @��\����s33@QG�B�RC@#�����!G�@UB��C<
                                    ByZX~  �          @��H���
=@P��Bp�C<�����=q@S33BC7�q                                    ByZg$  �          @��H��G��fff@W
=Bz�C@����G���@[�B z�C<                                      ByZu�  �          @���z=q���@c�
B%=qCC33�z=q�5@i��B*(�C>W
                                    ByZ�p  �          @�33��33���@O\)BCB�=��33�@  @U�B�C>L�                                    ByZ�  �          @��\����5@B�\B(�C<Ǯ�����
=@EB�
C9(�                                    ByZ��  �          @�  ��ff�aG�@:=qA�C6���ff=u@:�HA�Q�C3B�                                    ByZ�b  �          @�����ü�@>�RA�33C4O\����>��@>{A�=qC1�                                    ByZ�  �          @�(���{=��
@K�B33C3\��{>Ǯ@J=qB(�C/�=                                    ByZͮ  �          @������
�#�
@UB	��C4n���
>�\)@U�B	=qC0��                                    ByZ�T  �          @������H�u@W
=B�C6�����H=�Q�@W�Bp�C2��                                    ByZ��  �          @����p����R@P  B��C7�f��p�    @QG�BQ�C4�                                    ByZ��  �          @��R��  ��
=@Mp�B\)C8����  ��G�@O\)B�C5G�                                    By[F  �          @��R��=q��@EA�  C9&f��=q�#�
@G�A���C5��                                    By[�  �          @�����(���\)@H��A�\)C7
��(�<��
@I��A�ffC3�q                                    By[%�  �          @�\)����=�\)@@��A���C35�����>�Q�@?\)A���C0�                                    By[48  �          @�(���ff?k�@{A�(�C)����ff?�\)@�A�\)C'8R                                    By[B�  �          @��\���\@�=q?L��@�(�C����\@�(�>��@���C��                                    By[Q�  �          @�����(�@,��?�A��RCL���(�@7
=?��A�\)C��                                    By[`*  T          @����z�?s33@0��A�C(xR��z�?���@)��A��C%u�                                    By[n�  �          @�p���(�?fff@*=qA��C(�3��(�?�33@#�
A�C&
=                                    By[}v  �          @�Q���=q?5@>�RB(�C+(���=q?}p�@9��A�ffC'�=                                    By[�  �          @�����p�?z�H@R�\BQ�C&����p�?��
@L(�B��C"�H                                    By[��  �          @�����\?\(�@Y��B�C(�R���\?�
=@S�
B�\C$�                                     By[�h  �          @�{����%�@�
A��CP�����ff@#33A�(�CN��                                    By[�  �          @�p���
=��@K�A��HCN@ ��
=�G�@XQ�Bz�CK(�                                    By[ƴ  �          @�=q�����@  @4z�Aԣ�CT.�����.{@EA��HCQ�R                                    By[�Z  �          @˅�����c�
@B�\A�G�C[�
�����P��@W
=B {CYaH                                    By[�   �          @���~{��G�@'�Aƣ�Ca���~{�q�@?\)A��HC_��                                    By[�  �          @ʏ\��z��%?��RA��CM����z����@�RA���CK�                                    By\L  �          @�����H�{�@{A��C^&f���H�mp�@%�A�Q�C\z�                                    By\�  �          @ȣ��\)��G�@
�HA��\Cc��\)��=q@#�
A��Ca�{                                    By\�  �          @ʏ\���R���\?�33A�  Ca�{���R��(�@33A�z�C`}q                                    By\->  �          @�G���{����?�{A�{Ca�f��{���H@  A�z�C`T{                                    By\;�  �          @��H��Q���ff@ffA��RC`����Q��~�R@�RA�ffC_�                                    By\J�  �          @�(��������?�p�A��RCa@ �������@Q�A���C_޸                                    By\Y0  �          @�p������=q?�=qA�z�Cd
�����(�@  A�{Cb�H                                    By\g�  �          @���������(�?��A�=qCd��������{@{A�=qCc�=                                    By\v|  �          @�����z���
=?�z�A��C_޸��z�����@33A���C^��                                    By\�"  �          @θR�������?��HA�p�C^�)�����}p�@A�  C]33                                    By\��  �          @Ϯ�������R?���A�p�C^���������@\)A�=qC]��                                    By\�n  �          @�Q������{?�{A�  Cdff������@�\A��Cc33                                    By\�  �          @љ���(���
=?���A�(�Ca�
��(�����@ffA�ffC`G�                                    By\��  �          @�\)��ff��
=@�A�{C_�=��ff��Q�@{A��C^�                                    By\�`  �          @љ�������(�@�A�Q�C`ٚ������p�@�RA�{C_n                                    By\�  �          @�����
=��Q�@
=A���C_�
��
=����@\)A��C^�                                    By\�  �          @�
=��G����@(�A�=qC]�)��G��u@#�
A�Q�C\E                                    By\�R  �          @�����\)���@Q�A���C^z���\)�w�@0  A��C\Ǯ                                    By]�  �          @�
=���
���@   A��C^����
�s�
@7
=A�Q�C]!H                                    By]�  �          @�
=��33����@!�A�ffC^�3��33�s33@9��A���C]�                                    By]&D  �          @�  ��G���G�@.�RA���C_L���G��qG�@FffA�C]Q�                                    By]4�  �          @љ���  ��\)@(��A�  C`�
��  �~{@AG�Aۙ�C^�q                                    By]C�  �          @��
���R��G�@z�A�=qC_�f���R���@-p�A�p�C^L�                                    By]R6  �          @љ���{���@�\A�G�C_����{��Q�@*�HA�Q�C^\                                    By]`�  �          @�G������z�@A�{C^�����y��@.{Aď\C]�                                    By]o�  �          @������H����@33A���C]G����H�s33@*=qA��\C[�)                                    By]~(  T          @У���33�}p�@�A�Q�C\�q��33�n�R@.�RAŮC[                                      By]��  �          @љ����R��p�@�A�C_\���R�{�@0  A�z�C]aH                                    By]�t  �          @У���G���G�@
=A��Ca��G����@0  A�C_c�                                    By]�  �          @�33������@�A��C`+�����=q@.{A��HC^�\                                    By]��  �          @�33������Q�@�HA�ffC\Ǯ�����qG�@2�\A��
C[                                    By]�f  �          @ҏ\���\�p��@�HA���CY����\�aG�@0��AƏ\CX
                                    By]�  �          @ҏ\��G��u@�A��CZ� ��G��fff@.{A�\)CX��                                    By]�  �          @�33����|(�@ffA���C[Ǯ����mp�@-p�A��
CZ�                                    By]�X  �          @��
��p���  @�A���C\�=��p��p  @333A�{CZ�                                    By^�  �          @�(���p����H@�A�z�C]5���p��w
=@,��A�(�C[�=                                    By^�  �          @�(���  ����@�\A��C_����  ���@+�A��HC^�                                    By^J  �          @�(����R���H@\)A��RC`.���R���@(��A�(�C^�H                                    By^-�  �          @�  ��G��n{@G�A��CYٚ��G��`  @'
=A��CX)                                    By^<�  �          @����=q�fff@p�A�p�CX� ��=q�XQ�@"�\A��\CW                                    By^K<  �          @������H�s�
@
�HA��CZB����H�e@!G�A���CX��                                    By^Y�  �          @Ϯ�����o\)@  A�(�CY�������`��@%A�  CX0�                                    By^h�  �          @У���G��o\)@�A�{CY�q��G��`��@*�HA�  CX5�                                    By^w.  �          @�Q���{�g�@�A�
=CX@ ��{�Y��@ ��A�  CV��                                    By^��  �          @�  ��{�`  @
=A�ffCWL���{�P��@+�A¸RCUn                                    By^�z  �          @�����p��aG�@p�A���CW����p��QG�@1�A�G�CU�)                                    By^�   �          @�G����\�c�
@%�A�Q�CXh����\�S33@:=qA�G�CVaH                                    By^��  �          @У���33��33@%�A���C_G���33�u@<��A�C]h�                                    By^�l  �          @У�������R@'�A���C`�{����|(�@@��Aۙ�C^��                                    By^�  �          @�Q���{���@0  A�C`Ǯ��{�xQ�@HQ�A�C^�\                                    By^ݸ  T          @�Q���p����@5A��HC`����p��tz�@Mp�A�RC^z�                                    By^�^  �          @љ����\���@,(�A�C_xR���\�u@C�
A���C]�                                    By^�  �          @�Q���ff�}p�@$z�A��RC]����ff�l��@;�A�
=C[�                                    By_	�  �          @�  ��Q��|��@�RA�C]5���Q��l��@5A��
C[Y�                                    By_P  �          @Ϯ��G�����@(��A��\C_Y���G��q�@@��A��
C]ff                                    By_&�  �          @�
=���H����@%�A�=qC^�)���H�qG�@<��A�G�C\�3                                    By_5�  �          @�ff��ff�{�@8��A�\)C_���ff�h��@P  A�z�C\�                                    By_DB  "          @�p�����U�@(��A��
CV�R����Dz�@<(�A�  CT��                                    By_R�  �          @�����33�Mp�@ffA�CT.��33�@  @��A�z�CRn                                    By_a�  �          @�z���  �a�@��A��\CX����  �Q�@.{A�{CV�f                                    By_p4  "          @�p����
�xQ�@)��A�G�C]�����
�g
=@@��A�C[�\                                    By_~�  
�          @�p���p��g
=@ ��A�=qCY���p��W
=@5A�z�CW�                                     By_��  T          @�p���33�dz�@
�HA��RCXff��33�Vff@ ��A�=qCV�f                                    By_�&  T          @�\)��Q��J=q@(Q�A��CT5���Q��8��@:�HA�Q�CQ��                                    By_��  �          @θR���\�>�R@*=qA\CR\)���\�-p�@;�A�  CP\                                    By_�r  "          @�
=��33�333@4z�A�G�CPǮ��33�!G�@E�A㙚CNG�                                    By_�  
�          @�  ��z��HQ�@8��A��
CT�\��z��5�@K�A���CR\                                    By_־  
(          @�ff��p��Tz�@<��A�{CWn��p��AG�@P��A��HCT��                                    By_�d  
�          @�\)����Y��@E�A�
=CX�3����E@X��A���CV�                                    By_�
  T          @�z����aG�@<��A܏\CZu����N{@Q�A�\)CW��                                    By`�  T          @�=q��(��\��@=p�A�G�CZ@ ��(��I��@Q�A�  CW��                                    By`V  
�          @�=q��\)�'
=@W�B��CS����\)�G�@g
=B�CP5�                                    By`�  T          @���w
=�W�@7�A�=qC].�w
=�E�@L(�BffCZ�)                                    By`.�  �          @����l(��c�
@9��A�(�C_�q�l(��P��@N�RB33C]}q                                    By`=H  �          @��R�z=q�S�
@>{A���C\:��z=q�@  @Q�Bz�CY��                                    By`K�  �          @����Z=q���@)��A��Cf��Z=q�xQ�@B�\A�  Cd�R                                    By`Z�  
�          @���j�H����@p�A��Cc���j�H�qG�@5A�\Ca�                                    By`i:  
�          @�\)�G
=���R@{AÅCk{�G
=��ff@9��A�  Cin                                    By`w�  �          @�p��7
=���H@!�A��Cop��7
=��=q@?\)A�Q�Cm�R                                    By`��  T          @�p��,����\)@�HA�G�Cq�
�,����
=@9��A��CpG�                                    By`�,  T          @�G��333����@=qA�  Co��333����@7�A��CnE                                    By`��  �          @��H�+�����@33A��
Cr��+����\@"�\Ař�Cp�                                    By`�x  �          @��������(�@�
A�G�Ctp��������@#33A�  Csh�                                    By`�  �          @��H�����\)?��HA��RCuk������Q�@{A�Ctz�                                    By`��  �          @�G��G����\@Q�A�(�Ck� �G���=q@4z�A�p�Cj0�                                    By`�j  T          @��R�4z����@
�HA�  Co=q�4z���  @(Q�A��HCm�                                    By`�  T          @\�z�H��  @
=A��HCa�=�z�H�qG�@\)A�
=C_ٚ                                    By`��  �          @�33����]p�@A��CZ!H����Mp�@*�HAиRCX\                                    Bya
\  �          @��H�j=q�\)@.{A��Ccz��j=q�l��@FffA�Q�CaO\                                    Bya  �          @�33����QG�@�A���CW�����B�\@   A�\)CU{                                    Bya'�  �          @�z���G��H��@(�A�\)CU5���G��9��@\)A��CS33                                    Bya6N  T          @�33���H�:�H@��A��\CS{���H�+�@#33AŮCP�                                   ByaD�  �          @\�����=p�@ffA��CS�)�����-p�@(Q�A�\)CQ�H                                   ByaS�  �          @�����
�/\)@$z�A�G�CQ\)���
�{@5�A�33CN޸                                    