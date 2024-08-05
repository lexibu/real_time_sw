CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240720000000_e20240720235959_p20240721021525_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-21T02:15:25.525Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-20T00:00:00.000Z   time_coverage_end         2024-07-20T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy�   "          @�{?������H?
=q@�Q�C�>�?������?G�@�  C�Ff                                    By��  "          @�ff?��H��(�?�
=A`  C�*=?��H��=q?�A�ffC�7
                                    By� L  �          @��H?�ff���?�{AYG�C��=?�ff��  ?�AzffC��{                                    By�.�  �          @�{?�p����
?�z�A]p�C�=q?�p����?�33A~�RC�K�                                    By�=�  T          @���?�Q���  ?��
AH��C��?�Q���{?��
AjffC�\                                    By�L>  �          @�{?�p�����?�ffAN=qC�9�?�p����H?�ffAp  C�Ff                                    By�Z�  T          @�G�?��\�ָR?ٙ�A_�C�` ?��\��z�?���A���C�l�                                    By�i�  �          @�Q�?�p����?�\)Aw�C�9�?�p��ҏ\@�A���C�G�                                    By�x0  T          @ٙ�?�
=��  ?�
=AB�\C�:�?�
=��{?�
=Ad��C�H�                                    By׆�  
�          @ٙ�?�(���\)?�\)A]�C�XR?�(����?�\)A�C�e                                    Byו|  
(          @�=q?�����G�?�  A0(�C�0�?����Ǯ?�  AR�\C�=q                                    Byפ"  T          @љ�?�����H?Q�@�ffC�*=?���ə�?��A�C�33                                    Byײ�  �          @��?����33?n{A33C���?�����?�
=A&=qC�޸                                    By��n  �          @�\)?�  �Ǯ?��HA,Q�C��\?�  ��{?��HAO�
C���                                    By��  1          @�
=?��
����?���A���C��)?��
��ff@(�A�z�C��                                    By�޺  w          @���?�����?�G�A{C�!H?���Å?�G�A5�C�,�                                    By��`  
�          @ʏ\?�Q�����?��HA0  C��\?�Q����?���AS�
C��q                                    By��  �          @θR?�\)��p�?��A;
=C�C�?�\)���
?ǮA_33C�Q�                                    By�
�  
          @�?���?���AAG�C�Z�?����
?���Ae�C�g�                                    By�R  "          @�z�?aG���Q�?�A�Q�C�0�?aG���p�@
=qA���C�>�                                    By�'�  T          @�{?�Q���G�?�A���C���?�Q����R@33A�\)C���                                    By�6�  
�          @�\)?�R���
?�
=A���C��?�R��G�@
�HA�C�3                                    By�ED  �          @�Q�>8Q��\?�
=AS�
C���>8Q�����?�
=Az=qC��q                                    By�S�  �          @�=q>Ǯ�Å?�=qAhQ�C��{>Ǯ����?�A�p�C�ٚ                                    By�b�  �          @�{>��\@z�A��C�C�>����@z�A��C�L�                                    By�q6  
�          @�z�>���Q�@A�
=C�E>���p�@ffA��\C�N                                    By��  
�          @˅>.{��G�?��HA��C��>.{��ff@{A�p�C�Ф                                    By؎�  T          @�G�>.{��{@�\A�C��3>.{��33@33A���C���                                    By؝(  �          @��>#�
���H@�A�=qC���>#�
��\)@%A�Q�C���                                    Byث�  �          @˅>8Q�����@)��A��HC��>8Q�����@9��A�
=C��                                    Byغt  �          @�z�<��
��ff@2�\A��C�  <��
��=q@A�A�p�C�                                      By��  "          @��<���  @X��B�
C�33<���33@eB{C�5�                                    By���  "          @��>�����
=@7
=A�G�C�l�>������\@Dz�B�HC�~�                                    By��f  �          @�p��E��Tz�@o\)B>��C�w
�E��I��@xQ�BH�C��                                    By��  
�          @��
��z�˅@�Q�BvC_�R��z῱�@��\B}C[�R                                    By��  �          @�  ���?(��@�=qBu��C#�=���?Y��@���BrG�CJ=                                    By�X  T          @��\�=q?�(�@�Q�BP�\C�{�=q@	��@z=qBI�C
O\                                    By� �  �          @�p��  ?���@��
BgQ�C��  ?�\@�G�B`��C��                                    By�/�  
�          @����	��?���@�
=Btz�C���	��?�z�@��Bn�HC�                                    By�>J  "          @�z���\?Tz�@�z�B�(�C�f��\?�ff@�33B}��Cٚ                                    By�L�  
�          @��׿�ff?�ff@�ffB~
=CͿ�ff?�G�@�(�BwQ�C��                                    By�[�  	�          @���=q?�@��\By{C\��=q@�@�\)Bp��B�                                    By�j<  
i          @��H���R?�@��Bz�HCh����R?�33@��Bt�CO\                                    By�x�  
          @�=q��(�@"�\@��BZ{B�Q��(�@0  @�BPB�8R                                    Byه�  �          @�G���\@6ff@��HBJz�B󙚿�\@C33@|(�B@�HB�8R                                    Byٖ.  
�          @�Q��\)@Q�@\��B#�B�� �\)@\��@Q�Bz�B��f                                    By٤�  
�          @����G�@Mp�@eB-��B�aH�G�@X��@[�B#�B�                                    Byٳz  T          @�{�\(�?�p�@�(�B<=qCJ=�\(�?�
=@�G�B6�
C�H                                    By��   
          @����Z=q?��@�Q�BM�RC!}q�Z=q?�\)@�{BIC&f                                    By���            @����Vff@�@��B7
=C��Vff@��@�  B0z�C��                                    By��l  
�          @��S33?�z�@��RBOC �f�S33?���@���BK�\C8R                                    By��            @�33�I��?Tz�@���BY�RC%!H�I��?���@�  BVQ�C!=q                                    By���  "          @�p��L��?.{@�(�B[  C'�3�L��?k�@��HBX{C#�                                    By�^  "          @�  �QG�?#�
@���Bb  C(���QG�?fff@��B_=qC$�)                                    By�  �          @��HQ�u@�p�Bf�C5{�HQ�>L��@�p�Be�
C0Y�                                    By�(�  T          @��\)�޸R@�Bu
=CYٚ�\)���H@�Q�B|�CU+�                                    By�7P  
�          @�{��\�0��@��HBZ�Ci����\�   @�\)BdQ�Cf                                    By�E�  �          @��
��
=�7
=@��BW�HCk�R��
=�&ff@�z�Ba��Ciff                                    By�T�  �          @��
�Q����@�33B_�C_p��Q���R@�
=Bg�C[�H                                    By�cB  �          @�G��޸R�?\)@�z�BUQ�Co�{�޸R�.�R@���B`  Cm�=                                    By�q�  "          @����Q��0��@�z�B\ffCnz��Q��   @���Bg  Ck��                                    Byڀ�  
�          @�녿��0��@��\Ba�Cn�
���\)@�\)Bk�
Cl(�                                    Byڏ4  �          @�\)��p���H@�=qB{��Cs���p���@�ffB��Co�                                    Byڝ�  
�          @�p������@�p�BuQ�Co�R���
=q@���B�#�Cl�3                                    Byڬ�  
�          @�ff��=q��\@���Bx��Co�)��=q�   @�z�B�  Clh�                                    Byڻ&  "          @��
���
=@�  B|G�Ck녿�����@��B�ffCg��                                    By���  �          @��
��ff��G�@��HB�u�Cd�3��ff��(�@�B�33C_z�                                    By��r  
�          @��H��\)�+�@�z�Bf
=Ck(���\)�Q�@���Bp�Cg�f                                    By��  
�          @���33���@��Bk33Cg� ��33�
=@�{BuffCd                                    By���  T          @�{��=q�!G�@���BiQ�Cj{��=q�{@�{Bs��Cf��                                    By�d  
�          @������8Q�@��B^Q�Cm^����%�@���Bi�Cjs3                                    By�
  E          @Å��\)�5�@��\Ba{Cl���\)�!G�@��Bl=qCin                                    By�!�  
(          @�33�G���@��Be
=Cb@ �G���
@��Bn��C^=q                                    By�0V  T          @���-p���@�=qBc=qCWG��-p���{@�BjCR�                                    By�>�  �          @����G��.�R@�ffB^�Ci�{�G���@��Bi{CfJ=                                    By�M�  
i          @�ff���:=q@��HBW33Cp����'�@�Q�Bc(�CmxR                                    By�\H  �          @��R�����R�\@�\)BN�RCx+������@��@�B[�RCvW
                                    By�j�  T          @�(���(��\(�@��BNz�Cz����(��I��@�=qB[��Cxٚ                                    By�y�  T          @�{����n{@�  BE=qC~.����\(�@�
=BR��C|�                                    Byۈ:  �          @�Q쿊=q�{�@�p�B=(�C~�3��=q�i��@���BK
=C}��                                    Byۖ�  �          @�
=��p��x��@��B;��C|����p��g
=@�33BIffC{:�                                    Byۥ�  �          @�p���\)�z=q@�G�B:  C~  ��\)�hQ�@���BH  C|�
                                    By۴,  T          @�������=q@�p�B3=qC�
����r�\@�p�BAz�C~�
                                    By���  
�          @�녿У��P  @���BL��CsW
�У��<��@�\)BZ{Cq\                                    By��x  �          @��ÿ�=q�C33@���BV=qCr�׿�=q�/\)@��\BcQ�Cp�                                    By��  
�          @�zῠ  �A�@��\BY�RCw����  �.{@�Q�Bgz�Cu\)                                    By���  �          @�ff�#�
��p�@S33B�C��׽#�
��{@e�B"
=C��H                                    By��j  �          @��?��
��(�?���AjffC���?��
����?�ffA��HC��
                                    By�  
�          @�  ?����>�(�@z�HC��q?����(�?Q�@�=qC���                                    By��  �          @У�@���녾���33C�H�@��\�L�;�C�AH                                    By�)\  
�          @�ff@
�H�����W
=��{C��@
�H����>L��?�ffC��                                    By�8  �          @ƸR?�(����H?�@�  C�W
?�(���G�?uA\)C�k�                                    By�F�  �          @Å@  ��z�?�@�(�C��q@  ���H?uA��C���                                    By�UN  
�          @�{@%���=��
?G�C�ff@%���H>��@�(�C�o\                                    By�c�  �          @�  @Dz����5��G�C�4{@Dz����R���R�1G�C�!H                                    By�r�  
�          @θR@XQ����Ϳ�ff�p�C��@XQ����R�+���C��q                                    By܁@  
�          @�@P  ��Q������HC�C�@P  ��G��W
=����C�33                                    By܏�  1          @ȣ�@C�
��{�z�����C���@C�
��
=�B�\��(�C��)                                    Byܞ�  
�          @�33@^{���ÿ.{��p�C���@^{��녾����/\)C���                                    Byܭ2  �          @�(�@_\)��=q��33�K�C��q@_\)���H<�>�\)C���                                    Byܻ�  "          @�
=@C�
��(�?�@�C�˅@C�
���\?k�A��C��                                    By��~  �          @ƸR@*�H���?��A"=qC���@*�H����?��RA^ffC�(�                                    By��$  �          @Å@N�R��z�
=q���HC�{@N�R��p��#�
�\C��                                    By���  �          @�@W���{����Q�C�}q@W���>��@�C��H                                    By��p  
�          @ə�@aG���ff�L�;�C��@aG���{>�{@C�
C�3                                    By�  �          @ə�@e������Ϳk�C�Q�@e���p�>�z�@'�C�W
                                    By��  T          @ȣ�@`  �����H��Q�C��@`  ��ff���Ϳk�C���                                    By�"b  "          @��H@x����Q콸Q�Q�C��@x����  >�z�@'�C��3                                    By�1  �          @љ�@\)��{�u��\C��@\)��>�{@>{C���                                    By�?�  
�          @�Q�@xQ����R>W
=?�C�S3@xQ���?(�@�(�C�h�                                    By�NT  
�          @�
=@k���Q콸Q�Q�C�'�@k���  >���@0��C�,�                                    By�\�  �          @�33@]p�����>��?�\)C���@]p�����?�@��
C��H                                    By�k�  �          @�{@[����\>�
=@z=qC���@[�����?Q�@�z�C�                                      By�zF  
(          @Ǯ@k�����>\)?�  C�
@k���Q�?�@���C�*=                                    By݈�  T          @�{@�33���׿޸R�q�C�S3@�33��z῱��@z�C��=                                    Byݗ�  
�          @��@����y����
=���HC�q@�����G���{�`(�C��q                                    Byݦ8  c          @�  @���\)�	������C���@�����Ϳ����z�HC�                                      Byݴ�            @�Q�@�ff�z=q�����=qC���@�ff��33�z�����C�0�                                    By�Ä  
�          @���@�ff�qG������C��@�ff�|(���{�~�RC�p�                                    By��*  c          @ȣ�@�
=�L�������C��)@�
=�W���z����RC��\                                    By���  E          @��
@��H�HQ�� �����C���@��H�R�\�޸R���\C���                                    By��v  �          @�G�@��R�h�ÿ�����C�g�@��R�q녿�p��c
=C��                                    By��  �          @�  @��R�Tzῃ�
�-C�� @��R�X�ÿ@  ���
C���                                    By��  T          @��@����p��������
C�33@���У׿�(���=qC�J=                                    By�h  �          @��@s�
�Q녿�(��ָRC��{@s�
�z�H������
C���                                    By�*  T          @�{@�
=����
=q��{C�(�@�
=�\��\��p�C��                                    By�8�  �          @���@z���>��@�{C���@z����>��@��C�                                    By�GZ  �          @s�
����*=q@\)B+C�(�����(�@-p�B?z�C��f                                    By�V   
}          @`��>#�
�L(�?�
=A��\C�n>#�
�C33?ٙ�A�ffC�~�                                    By�d�  �          @o\)���'
=@'
=B2�C������@5�BF=qC��=                                    By�sL  T          @�Q���(�@G�BN  C������
=q@U�Ba��C���                                    Byށ�  T          @a녾W
=�Q�@!G�B:{C��f�W
=�	��@.{BNQ�C�B�                                    Byސ�  
�          @Dz�>.{�*�H?�{A��
C�ٚ>.{�#�
?��Aܣ�C���                                    Byޟ>  T          @>�R�W
=�ff?�B{C�|)�W
=�
�H@ ��B*��C�H�                                    Byޭ�  �          @J�H��(���@ ��B��C��3��(��
�H@{B4(�C�b�                                    By޼�  
�          @Tz�=u�;�?�  A���C�� =u�2�\?�G�BQ�C���                                    By��0  "          @\(�=��
�;�?޸RA�p�C�Ф=��
�0��@   B�C��q                                    By���  
�          @dz᾽p��%�@(�B,ffC��׾�p��@*=qBA=qC�|)                                    By��|  "          @e��
=q��
@)��B@��C�c׿
=q��
@7
=BU=qC5�                                    By��"  T          @[��B�\��R@\)B#\)C}��B�\���@p�B7�C{xR                                    By��  �          @:�H����?�G�B��CT{����{?�Q�B4�C~                                    By�n  T          @aG�=#�
���@*=qBI��C��H=#�
��Q�@7
=B_33C��3                                    By�#  
�          @G���33��ff@:�HB��fCg�ᾳ33�.{@<��B��HCM�{                                    By�1�  "          @,�Ϳ�\?
=?�p�B��RC�)��\?G�?�z�B��B��q                                    By�@`  "          @|(����R@U�������{B�
=���R@Mp�����33B�u�                                    By�O  "          @8Q�}p�@)��>���@��RB�\�}p�@*�H>\)@1�Bܳ3                                    By�]�  "          @;��n{@,(�?
=qA+33B�#׿n{@.�R>�=q@�Bٞ�                                    By�lR  
�          @>�R�fff@333>�33@�(�Bה{�fff@4z�=��
?��B�Q�                                    By�z�  
�          @QG��W
=@?\)?�AffB�ff�W
=@A�>k�@�
=B�
=                                    By߉�  "          @AG��&ff@:=q>�@%�B�\�&ff@:=q����<��B�\                                    ByߘD  "          @H�ÿxQ�@8�ÿ=p��Yp�B��xQ�@333���\���B��                                    Byߦ�  �          @:=q�:�H@-p��&ff�O\)B�B��:�H@(Q�h�����B��                                    Byߵ�  �          @;���=q@
=��  ��HB�ff��=q?������4�RB�Ǯ                                    By��6  
�          @Fff��@ff����<  B�aH��?�{�p��S(�B�33                                    By���  �          @N�R�\)@{�z����B��{�\)@\)��
�7=qB�=q                                    By��  
Z          @K�?G�>���3�
#�A�?G�>8Q��5�{AM�                                    By��(  
Z          @L(�>��?aG��8���Bv�>��?��=p�ǮBJ(�                                    By���  
�          @L��>B�\@�Ϳ������
B�33>B�\@33�Ǯ���B���                                    By�t  T          @Y����@�׾����   B��;�@�Ϳ!G��\)B���                                    By�  
�          @G
==u?�(��z��FG�B���=u?�(�� ���^G�B��                                    By�*�  �          @S�
<#�
?\(��J�H��B�\<#�
?��O\)¡�)B���                                    By�9f  T          @N{��z�?333�E�L�B���z�>�p��HQ�¢�HC                                     By�H  
�          @A녾\?#�
�7��RB񞸾\>�{�;� 8RC
�f                                    By�V�  
Z          @E=u?0���?\)B�#�=u>�p��B�\¥��B���                                    By�eX  "          @E=�?�33�5���B�L�=�?W
=�<(��B�                                      By�s�  T          @N{=�G�>����J�H¥(�B��R=�G�=L���L��¯� A�=q                                    By���  
�          @N{�:�H�u�E�L�C9�:�H�����C�
��CP�R                                    By��J  
�          @S33�xQ�����Fff�=CJff�xQ�=p��B�\=qCY0�                                    By���  
�          @
�H�u�Tz���S�C\�
�u��G������Bp�Cbu�                                    By஖  
�          @(���
=��녾�G��G�C`0���
=��
=��\)� (�Ca�                                    By�<  �          @ �׿��H��
=�����Q�C`Q쿚�H����n{��{Cb�f                                    By���  
�          @'���{��
=��{�6�ChaH��{�У׿�Q��"
=Ck��                                    By�ڈ  
�          @N{��\>����.{
=CLͿ�\<��
�/\)
=C2\                                    By��.  
�          @XQ�?z�?��H�5�fz�B���?z�?�\)�@���~��B�z�                                    By���  "          @u?��@
=�0���9�B��{?��@��@���Q�Bs{                                    By�z  
�          @}p�?���@"�\�0  �.��Bx=q?���@p��AG��E��Bj�                                    By�   �          @��H?�Q�@
=q�L(��Q{BtQ�?�Q�?��
�Z�H�g��B`�\                                    By�#�  "          @k�?�ff@
=q�+��:�Bl  ?�ff?��:=q�Q(�BZ��                                    By�2l  
(          @�  ?���?���N{�Y��B[p�?���?��R�Z�H�n�BA��                                    By�A  �          @���?W
=@   �XQ��e
=B�W
?W
=?˅�e�}Q�Bx�                                    By�O�  "          @��H?�  @��Y���O{Bx\)?�  ?����j=q�fffBe{                                    By�^^  �          @�{?Tz�@ff�Tz��T��B��f?Tz�?�Q��e��m�HB�z�                                    By�m  �          @dz�?L��?޸R�*�H�YQ�B���?L��?�z��7
=�q��Bq                                    By�{�  
(          ?����L��?����
��B�\)�L��?�z�aG���  B�q                                    By�P  
�          @z῀  ?�=�Q�@(�B��
��  ?��\)�j�HB��f                                    By��  	�          @6ff?.{@zῸQ���=qB�{?.{@Q���H���B�Q�                                    By᧜  
}          @33?.{?�p���=q�(�B��R?.{?�ff���
�'�HB�ff                                    By�B  �          @(Q�?n{?��
��
=�ffByp�?n{?��ÿ���7�
Bl�R                                    By���  T          @r�\��
=?��@*=qB@�HB�.��
=@p�@=qB)��B�                                      By�ӎ  T          @w
=�8Q�@L��?�A�G�B�uÿ8Q�@Y��?�33A�  B�                                    By��4  	�          @z�H>#�
@w�?�\@�=qB�\)>#�
@y��<�>��B�ff                                    By���  �          @���?�@\)�#�
���B�?�@|�Ϳ
=q��B���                                    By���  T          @|(�?
=q@j=q��G���=qB��?
=q@^{���H��
=B�8R                                    By�&  "          @�p����
@��ÿ����\B������
@qG���
��ffB��3                                    By��  T          @�ff<#�
@�ff��\)����B���<#�
@�  ������HB���                                    By�+r  
�          @���u@�
=������B��׽u@|���
=��33B�                                    By�:  �          @����L��@����=q�\��B�33�L��@�������
=B�k�                                    By�H�  "          @��
�B�\@�  ��ff�k33B�z�B�\@u��Ǯ���\B��R                                    By�Wd  �          @c33���H@I����ff���Bų3���H@:�H��Q���
B�
=                                    By�f
  �          @c33��Q�@N{��p���Q�B�녾�Q�@@  �����p�B��
                                    By�t�  "          @g����R@X�ÿz�H����B�zᾞ�R@N�R����ffB�                                      By�V  	o          @}p��h��@tzᾨ����G�B��
�h��@p  �Tz��A��B�\)                                    By��  
}          @xQ��@s�
���H��Q�B���@mp��}p��k�
B�\)                                    By⠢  �          @z=q��ff@w
=�Ǯ��Q�B�=q��ff@q녿fff�T��B��=                                    By�H  
(          @�z��@�=q�������B�  ��@���fff�>{B�L�                                    By��  "          @�33��@�p��^�R�1�B�L;�@��׿������B�k�                                    By�̔  T          @����u@���    �uB̊=�u@��������HB�                                    By��:  "          @��aG�@��\�#�
�G�B��
�aG�@��׿G��{B�#�                                    By���  �          @�z�k�@���>#�
?�
=B�Ǯ�k�@��þ���ffB��H                                    By���  �          @��Ϳ��@�  >��?�
=B��f���@�\)��(���p�B�                                    By�,  
�          @��R�
=@���=u?J=qB�LͿ
=@��
�
=q��Q�B�k�                                    By��  
�          @�=q��=q@��H?J=qA(��B�p���=q@��>aG�@9��B���                                    By�$x  �          @��ÿ���@j=q>��@(�B�LͿ���@h�þ�33���\B�u�                                    By�3  �          @z�H��  @J�H?�G�A��RB�{��  @S�
?O\)AI��B��                                    By�A�  �          @c�
��{@5?�=qA��Bި���{@C�
?�A��B�                                    By�Pj  �          @]p��z�H@@��?�z�A���B��f�z�H@K�?xQ�A�(�B��                                    By�_  T          @Z�H���@@��?�(�A�33Bڽq���@I��?G�AS�
B��                                    By�m�  �          @Z�H��Q�@AG�?�ffA�\)B��Ϳ�Q�@H��?(�A%G�B�W
                                    By�|\  �          @Vff��(�@@  ?Q�Ac�B�k���(�@E�>Ǯ@���B�L�                                    By�  �          @o\)��z�@J=q?}p�Ax  B�aH��z�@QG�?�A33B���                                    By㙨  
�          @���ٙ�@hQ�?s33ATQ�B���ٙ�@n�R>��@���B��                                    By�N  T          @~{��33@b�\?:�HA)p�B�#׿�33@fff>W
=@Dz�B�L�                                    By��  
�          @w
=��(�@XQ�?��A�RB�{��(�@[�=���?��RB�ff                                    By�Ś  �          @e�   >���@��B$Q�C,h��   ?&ff@Q�B�HC%p�                                    By��@  �          @����!G���
=@3�
B+p�CU��!G���  @AG�B<\)CN}q                                    By���  �          @�ff����   @.�RB$�C[Ǯ�������@?\)B8�CUJ=                                    By��  T          @�G����h��@��A�G�C��
���QG�@-p�B(�C��f                                    By� 2  �          @xQ��Q���@��B��C_���Q��33@.{B5G�CY�                                     By��  T          @����K���G�@�B�HCMc��K����@$z�B
=CG��                                    By�~  T          @�  �9����Q�@&ffB&��C;��9��<�@(Q�B(��C3c�                                    By�,$  �          @�p����� ��@   B��Ca����
=@6ffB%�C\                                    By�:�  "          @�{�7���
@:=qB��CW�H�7��˅@L(�B0��CQ�                                    By�Ip  T          @�33�`��>�z�@O\)B*\)C/W
�`��?E�@J=qB%(�C'��                                    By�X  
(          @�\)�n�R?��H@&ffA�33C8R�n�R@@G�A�(�C�                                    By�f�  	�          @���^{@ff@�\A�
=C�
�^{@�?�Q�Aď\C{                                    By�ub  �          @��\�Z�H@	��@
�HA�ffCٚ�Z�H@p�?���A��HCJ=                                    By�  "          @��
�Vff@p�@�
A�G�C�)�Vff@"�\?�Q�A�(�C�)                                    By䒮  "          @��R�_\)@
=@ffAӮC��_\)@*=q?��HA�  C��                                    By�T  
(          @��w
=@(�@�HA��C���w
=@2�\@G�A��C+�                                    By��  T          @�p��~�R@7�@33A�33C@ �~�R@L(�?�=qA���CJ=                                    By侠  T          @��
����@P  @(�A�\)C!H����@c33?�33A��RC
�)                                    By��F  T          @�\)��G�@/\)@{AָRC�)��G�@Fff@ ��A�(�C��                                    By���  T          @��\)@7
=@�\A�(�C\)�\)@K�?�A�{CaH                                    By��  �          @��
��G�@:�H@   A���C)��G�@L��?�G�A��\C�{                                    By��8  T          @�����@7
=@ ��A���C�
���@H��?��
A��HC=q                                    By��  T          @�  ���@:=q@{A��\C�����@N�R?�p�A�Q�C�{                                    By��  T          @�p���  @AG�@p�A���C�{��  @U?�Q�A��Cٚ                                    By�%*  "          @��R����@/\)@��A�  C\����@E?�z�A���C��                                    By�3�  
�          @�ff���\@>{@{A�
=C�����\@R�\?ٙ�A��C�                                     By�Bv  T          @�=q��(�@<(�@p�A�=qC���(�@S33?���A��\C�R                                    By�Q  
�          @��
��ff@3�
@&ffA�ffC����ff@L��@ffA�{C@                                     By�_�  �          @�����=q@'
=@3�
A���C�{��=q@B�\@A�G�C޸                                    By�nh  
�          @�{��{@�R@:=qA�\)C^���{@;�@p�A�  C�                                    By�}  "          @���x��@(�@C33B	ffC�3�x��@*=q@)��A�ffC��                                    By勴  "          @�=q��Q�@
=@O\)BQ�C:���Q�@'�@5A���C�\                                    By�Z  
�          @������@�@6ffA��C����@4z�@=qAΏ\C�)                                    By�   T          @������@�@@  B�\C�����@5@#�
A�
=C�\                                    By左  
(          @����y��?���@]p�B�C���y��@Q�@FffB��C��                                    By��L  T          @���|��?�@^{B\)C\)�|��@�@G
=B{C                                      By���  T          @��\�~{?�\)@\��B��CǮ�~{@�@E�B  C�=                                    By��  T          @���z�H?��@\��B��C\)�z�H@(�@E�B�\C�                                    By��>  
Z          @�\)�n�R?�Q�@h��B&�C�f�n�R@�\@S33B��Cz�                                    By� �  �          @�\)�aG�?���@x��B5(�C���aG�@{@c�
B"\)C�=                                    By��  "          @��H�`  ?��R@��B;��C޸�`  @
�H@o\)B)�C!H                                    By�0  
�          @����a�?�
=@���B:ffC���a�@
=@n{B(z�C)                                    By�,�  
�          @�p��p��?c�
@��B;
=C&�p��?˅@x��B.�C�                                    By�;|  	�          @�G��qG�?�  @���B>�C%#��qG�?�p�@���B0��CO\                                    By�J"  T          @�=q�z�H?u@�B7�C&:��z�H?�
=@|(�B+  C�\                                    By�X�  �          @��
�y��?��H@�(�B3(�Cs3�y��@
�H@s�
B"  C�                                    By�gn  �          @��
�~�R?Ǯ@���B-  C�{�~�R@  @k�BffC��                                    By�v  �          @�33�}p�?�=q@�=qB1  C!ff�}p�@�\@qG�B!  C��                                   By愺  "          @����\)?��R@���B/ffC"���\)?�Q�@n�RB (�C�                                   By�`  "          @�Q��tz�?�
=@��B3p�C��tz�@Q�@o\)B"  C�
                                    By�  �          @���s�
?�ff@x��B-z�C�H�s�
@{@c33B{C�                                     By氬  "          @�p��n�R?�G�@xQ�B-
=C�3�n�R@(�@`  B��C�)                                    By�R  
Z          @��R�tz�?�z�@r�\B&p�Cp��tz�@$z�@XQ�BQ�C�                                    By���  �          @�p��w
=@   @h��B�RC���w
=@'�@Mp�B
33C�=                                    By�ܞ  �          @��u@33@h��B��C���u@,(�@Mp�B	��C�q                                    By��D  �          @�{�n�R@�
@g�B{C0��n�R@<(�@HQ�B�RC�                                    By���  �          @�  �l(�@%�@c�
BG�C
=�l(�@L(�@A�A�=qC#�                                    By��  �          @��dz�@R�\@;�A�
=C	T{�dz�@qG�@G�A�
=Cc�                                    By�6  �          @�33�Z�H@c�
@H��B�
C�
�Z�H@��\@�A��C��                                    By�%�  "          @�(��a�@g�@@  A��CE�a�@��
@�A�=qC�)                                    By�4�  �          @�
=�h��@l��@<��A�\)C�=�h��@�@p�A�  C�                                    By�C(  �          @��\�4z�@p  @\(�B  B�Ǯ�4z�@��\@*�HA�
=B�{                                    By�Q�  �          @�{�Fff@l��@]p�Bp�C�f�Fff@�G�@,��A�=qB���                                    By�`t  
�          @�G��P  @��@�A��B����P  @���?��Aw\)B�B�                                    By�o  "          @���U�@�ff@\)A�\)C ff�U�@���?�33A_\)B�ff                                    By�}�  "          @��\�W
=@��\@p�A�p�Cz��W
=@�
=?У�A�Q�B��
                                    By�f  
�          @����U@y��@.{A�G�C�=�U@�33?�A��RB���                                    By�  
�          @�(��Y��@q�@�A�(�C���Y��@�?�z�A���C#�                                    By穲  �          @�{�W�@l(�@.�RA���CxR�W�@�z�?��HA��C+�                                    By�X  �          @��O\)@l��@4z�A�\C5��O\)@�@33A���B���                                    By���  �          @�\)�J�H@aG�@L��B33C�J�H@��\@p�AˮB���                                    By�դ  
(          @���C�
@u@Mp�BC ���C�
@�z�@��A�\)B��3                                    By��J  
�          @��\�A�@���@;�A�p�B���A�@���@A��
B���                                    By���  T          @�G��=p�@��@.�RA�G�B����=p�@��
?�{A�(�B�G�                                    By��  
�          @���C�
@�Q�@&ffA���B�L��C�
@�{?��HA��B�33                                    By�<  �          @�(��AG�@�p�@�AÅB����AG�@��?�G�An�\B�Q�                                    By��  �          @�(��.{@��@�HA��HB�  �.{@�  ?��HAfffB�                                    By�-�  �          @�
=�,��@�  @A�(�B��,��@��
?�33AaG�B�
=                                    By�<.  
�          @�Q��@  @��@ ��AŮB����@  @�
=?�ffAo\)B�33                                    By�J�  �          @�\)�@  @���@��AŅB����@  @�p�?�  AqG�B�L�                                    By�Yz  �          @�Q��(��@�{?h��A\)B� �(��@���    �#�
B�z�                                    By�h   
�          @�p��>{@�p�@��A��
B����>{@�=q?�  Au��B��                                    By�v�  
�          @����K�@u@<��A�33C���K�@��@�A�\)B�(�                                    By�l  T          @���G
=@Tz�@QG�B�C&f�G
=@z�H@!�AָRC z�                                    By�  �          @�G��AG�@[�@HQ�B	��CT{�AG�@�  @
=A�{B��                                    By袸  �          @��:=q@S33@+�A���Cn�:=q@qG�?���A��B�B�                                    By�^  
�          @��R��
@X��@�A���B�=��
@n�R?��A��HB��q                                    By��  T          @�z��   @`��?��A�z�B�k��   @q�?aG�A:=qB��                                    By�Ϊ  
�          @�����@C33@	��A���B�� ��@[�?�p�A��HB�\)                                    By��P  �          @�=q��(�@R�\?���A��B�ff��(�@c33?c�
AK33B��)                                    By���  
�          @�  ��
=@aG�?�A��HB�LͿ�
=@k�>��@�ffB�z�                                    By���  �          @z�H���@a�?��
As�B݊=���@j�H>�=q@z�HB��                                    By�	B  "          @o\)���\@^{?   @�ffB�Q쿢�\@_\)�u�j=qB�                                    By��  
(          @`  ��{@S�
���ͿУ�B�#׿�{@Mp��J=q�QB�#�                                    By�&�  
�          @H�ÿ^�R@?\)�.{�HQ�B�z�^�R@8Q�L���l��BՏ\                                    By�54  T          @aG��0��@J�H��ff���\B�ff�0��@@�׿�{��  B͸R                                    By�C�  T          @��þ8Q�@�  ����33B�{�8Q�@aG��(����\B�Ǯ                                    By�R�  �          @�p����\@p  ��\)����B�p����\@QG��'���RB֮                                    By�a&  
�          @�G��.{@G
=�`  �A33B�G��.{@�\���\�rp�B��                                    By�o�  T          @�z����@>�R�n�R�MffB��q����@
=�����~�HB�G�                                    By�~r  �          @�ff?+�@=q��\)�mp�B��?+�?��������B�B�                                    By�  T          @���?\)@8�����H�Y
=B��?\)?�Q����B��)                                    By雾  
�          @��?�=q@G�����s(�B���?�=q?��R���
�BC��                                    By�d  �          @��
?��
@33��ff�q�B���?��
?��
��33�BLQ�                                    By�
  
�          @���?�@33��=q�z�\Bq\)?�?�G�����
=B#=q                                    By�ǰ  T          @�ff?�
=?�
=����y��BUp�?�
=?aG���p�B�A�G�                                    By��V  "          @�z�?�33?ٙ����H�B7(�?�33?
=���H  A��\                                    By���  T          @�{?�=q?��H��\)� B�R?�=q=��
���
�@%                                    By��  
�          @�  ?�?0����33Q�A���?���ff���
�HC�p�                                    By�H  T          @�33@�\>\)���R.@��@�\��G����
C���                                    By��  
�          @�ff@Q�<#�
�����>��@Q쿕����ffC��\                                    By��  T          @�{?�Q�W
=���C�޸?�Q쿫���z�z�C���                                    By�.:  
�          @�(�@�\�&ff��p���C�,�@�\��p������pffC��
                                    By�<�  �          @��@�\���R���
�rQ�C��@�\�\)��
=�S=qC�Ф                                    By�K�  �          @�G�@
=q��Q���=q�q�RC�  @
=q����(��O33C��=                                    By�Z,  "          @���?�33���������p��C��?�33�.�R�p  �F�C���                                    By�h�  �          @�����?��?���B*��C�3����?��?�G�Bp�B�{                                    By�wx  T          @����p�@@��@��
BKz�B�\)��p�@w�@U�B\)B��H                                    By�  "          @�ff��G�@e@��B;��B�uÿ�G�@�ff@O\)B
\)B�k�                                    By��  �          @��H���@Tz�@$z�B�B�LͿ��@tz�?�  A�(�B��                                    By�j  �          @��ÿ��H@}p�@^�RB�
B܅���H@���@�RA�p�B��f                                    By�  "          @����Q�@��@H��B��B�
=��Q�@���@ffA�
=B��                                    By���  
�          @�z῞�R@s33@L��B��B�����R@�@\)A�\)B�8R                                    By��\  �          @�Q�@  @Tz�@j=qB<p�B�p��@  @��\@333B�B��
                                    By��  T          @��>�@�@~{Bn{B���>�@L(�@Tz�B8��B�Q�                                    By��  T          @w�>���?Ǯ@`  B��=B�u�>���@�@B�\BPz�B���                                    By��N  �          @7�?�z��=q>\A
=C�?�z��Q�?L��A��HC�.                                    By�	�  T          @:=q@�Ϳ�33?.{AY�C�q@�Ϳ���?}p�A��
C���                                    By��  "          @+�@"�\�   ?�RAU�C�z�@"�\��{?5Ay�C�5�                                    By�'@  "          @A�@@  ��Q�>aG�@�(�C���@@  ����>���@�(�C�&f                                    By�5�  
�          @K�@J�H�������0��C��q@J�H��zὸQ�ǮC�Z�                                    By�D�  
�          @L(�@J�H<#�
��z����>�  @J�H�u��z�����C��                                     By�S2  �          @L(�@>�R���h����  C�ff@>�R�.{�E��aC���                                    By�a�  T          @L(�@Q쿆ff�G��!�C��{@Q��  ��(����C�]q                                    By�p~  �          @,(�@zὣ�
��z��ԣ�C��R@zᾸQ쿎{��=qC��q                                    By�$  "          ?�p�?�(�>�z�k���\)A�?�(�=u�xQ���ff?�                                    By��  
�          @Q�?��?s33�u��
=Aՙ�?��?0�׿�z���\)A��                                    By�p  T          @B�\?�G�@0�׾\��z�B��?�G�@%���
��\)B�z�                                    By�  
�          @>�R?c�
@*�H�\��(�B�\)?c�
@   ��G�����B�\                                    By빼  T          @�{?���@s33��Q����HB�aH?���@Y��������B���                                    By��b  "          @�33?��@�\)�\)�У�B�#�?��@�녿��
�u�B�8R                                    By��  �          @�z�?!G�@�=q>aG�@"�\B�  ?!G�@�\)�u�6�\B��3                                    By��  "          @�33?5@��þ\���B�?5@�G������{B�#�                                    By��T  �          @�\)?+�@�p����Ϳ�B�Ǯ?+�@�Q쿞�R�r�RB�33                                    By��  T          @�p�>�ff@�(�>�\)@g
=B�33>�ff@�녿L���(  B�                                    By��  
Z          @�=q>�  @���>\@�p�B���>�  @\)��R�z�B�                                    By� F  "          @�����@���?
=Az�B�L;��@�����
=��ffB�B�                                    By�.�  �          @��H�fff@z�H?(��AG�B��Ϳfff@}p��������
B͊=                                    By�=�  T          @�p��G�@���?!G�A  BɸR�G�@��\�Ǯ���HBɏ\                                    By�L8  	�          @�{�W
=@�G�?(��AG�B�ff�W
=@��\��p���Q�B�33                                    By�Z�  
�          @�=q���H@�Q�=�G�?��RB�{���H@��Ϳs33�Mp�B�k�                                    By�i�  v          @�?p��@�\)������(�B�G�?p��@~�R�!G����B�aH                                    By�x*  �          @�z�@G
=@��]p��*ffB	\)@G
=?�Q��xQ��E��A�p�                                    By��  �          @�{@1�@P  �W
=��RBE��@1�@33��=q�A��B��                                    By�v  "          @�=q@��@S33�Z�H� 
=BX(�@��@������Lp�B1                                      By�  �          @��
@   @@���k��,�
BI  @   ?�p���=q�V
=B��                                    By��  "          @���@\)@A��k��,�RBJ�@\)@   ���\�VQ�BQ�                                    By��h  �          @�G�@/\)@0  �Z=q�%ffB433@/\)?����  �J��B��                                    By��  
�          @��@\(�?��w
=�2�RA�\)@\(�?L����ff�GAQp�                                    By�޴  �          @�Q�@Z�H?���s�
�:G�A�33@Z�H>��~{�E  @
=q                                    By��Z  �          @��
@QG�?����p  �=
=A��@QG�>��z=q�H(�@33                                    By��   T          @�=q@`  ?�p��\(��+\)A�Q�@`  >����h���8  @�{                                    By�
�  T          @��\@n{@���C�
��A��
@n{?���`���&�HA��R                                    By�L  �          @�p�@h��?�Q��W
=�"��A��R@h��?��g��2\)A33                                    By�'�  �          @��@u�?=p��g��+�A/�@u��aG��l(��/��C�^�                                    By�6�  T          @��\@]p�?L���}p��@G�AP  @]p���  �����D��C��=                                    By�E>  T          @���@E�?��R����L(�A��@E�>���G��Y�@�R                                    By�S�  
�          @��
@o\)>��r�\�4@�{@o\)����q��4  C���                                    By�b�  �          @�{@n{?z��w��733AG�@n{��ff�x���833C���                                    By�q0  T          @�  @j�H?���z�H�6
=A�z�@j�H=��
���\�@  ?�                                      By��  �          @�p�@E�@!G��k��+{BQ�@E�?�(����R�L
=A�{                                    By�|  T          @��@?\)@2�\�p  �)�B,�@?\)?��H��33�Nz�A�                                      By�"  T          @���@3�
@:�H�p  �+  B8
=@3�
?�=q��(��R=qB�H                                    By���  �          @�{@Fff@<���p  �$��B.=q@Fff?��������JffA���                                    By��n  
�          @�ff@E@8���r�\�'z�B,33@E?��
��p��LffA                                    By��  T          @���@J�H@?\)�c33��B-�\@J�H?�Q���
=�Bp�A�ff                                    By�׺  T          @�(�@N�R@\)�����6��B  @N�R?����\)�RA�33                                    By��`  
�          @�33@X��@ ���~�R�5
=A��R@X��?^�R���
�M{Af�R                                    By��  �          @���@fff?�\)�o\)�*�RA�  @fff?O\)����@G�AI                                    By��  
�          @�Q�@\(�?����i���0
=A�(�@\(�?��|(��B(�A��                                    By�R  �          @���@g�?�Q��aG��(G�A�33@g�>�G��q��7�H@�                                    By� �  T          @�  @|��?����e��"�A���@|��>����s�
�/z�@��\                                    By�/�  
�          @�G�@�
=?��fff��RA�
=@�
=?#�
�z�H�*=qA	�                                    By�>D  T          @�@��?�(��U��z�Az�\@��>�=q�b�\�{@fff                                    By�L�  T          @���@�ff�k��8�����\C���@�ff��  �.{��(�C��R                                    By�[�  �          @�p�@����=q�/\)��G�C�]q@����  �#�
��=qC��                                    By�j6  T          @���@�G���
=�'
=��{C�T{@�G���\)������HC�{                                    By�x�  �          @�p�@�G��   ����܏\C���@�G�����
=q��  C���                                    By  �          @�G�@����
�Q���G�C��@��u�������C��                                     By�(  �          @�z�@�(����
�*�H��=qC��@�(������R����C�XR                                    By��  
�          @�{@��H=L���������?(��@��H�(��z���G�C�8R                                    By�t  
�          @�
=@�=q>\������  @�
=@�=q�8Q��p�����C���                                    By��  �          @���@���>���{��p�@�{@��ü#�
����33C��R                                    By���  T          @�  @�G�?z��z���  @�ff@�G��#�
�	�����C��                                    By��f  �          @��
@��>��
�&ff���@{�@������%���\)C�xR                                    By��  �          @�G�@�=q��G��Fff�{C�>�@�=q���
�6ff��(�C�&f                                    By���  �          @���@��
���A���HC�1�@��
�xQ��7���z�C��                                    By�X  "          @�@��H?!G��3�
��Q�@�(�@��H�8Q��8Q�� ffC��f                                    By��  �          @�Q�@��
?�  �5���
=AC�
@��
>.{�@  ��R@Q�                                    By�(�  T          @�ff@��\?��R�hQ���\A���@��\>���y���.Q�@�G�                                    By�7J  �          @��R@�p�?����C�
���AY��@�p�>k��P  �\)@0��                                    By�E�  "          @���@��>.{�H����Q�?�\)@���8Q��C�
���C��R                                    By�T�  
�          @Ǯ@��R�E������ ��C�]q@��R��\)�i�����C�'�                                    By�c<  �          @���@�Q�?�  �j=q�G�AHz�@�Q콏\)�r�\� 
=C���                                    By�q�  �          @�ff@��\?O\)��  �)�A(��@��\��Q���=q�,�C��                                     By  T          @�\)@��R?:�H�P  �A{@��R�aG��Tz��(�C��R                                    By�.  �          @�p�@��>\�6ff��\@�@����ff�5�噚C��                                    By��  "          @�
=@��\>�=q�B�\� \)@K�@��\�(��?\)��{C�j=                                    By�z  "          @��\@�z�?���p���=qA�\)@�z�?���(Q���  Ak�
                                    By�   �          @��\@tz�@\�Ϳ����p��B(ff@tz�@:=q��
���B=q                                    By���  �          @���?^�R@���?��@�z�B��?^�R@��׿&ff�33B�{                                    By��l  �          @�=q��p�@�ff@�A��HB܅��p�@�ff?xQ�A!G�B��f                                    By��  
�          @������@�G�@#�
Aԣ�Bݔ{����@�=q?�=qA-�Bٳ3                                    By���  "          @��׿�33@���@(��A�G�B�LͿ�33@��\?�33A7
=B�#�                                    By�^  "          @����\)@���@4z�A�  B�R�\)@�z�?�33A_�
B�\                                    By�  "          @�Q��ff@���@8Q�A�=qB����ff@��?���Ah��B�B�                                    By�!�  "          @�ff�<(�@�Q�@,(�A�\)B��\�<(�@��
?�z�Ag33B�                                    By�0P  �          @���@�ff@=p�A�ffB�����@�(�?�{A�Q�B�Ǯ                                    By�>�  �          @�ff�{@�\)@.{A��B�R�{@��\?�ffAS33B�33                                    By�M�  �          @�p�����@�33@(�A�=qB߽q����@���?.{@��B�z�                                    By�\B  �          @�z���
@�ff@33A�(�Bۀ ���
@��\?�\@�B��)                                    By�j�  �          @�����
@��?�\)A�=qB����
@�=q=u?#�
B�
=                                    By�y�  T          @��׿�
=@�\)?�A�\)B�8R��
=@�  >\)?�
=B�k�                                    By��4  �          @�G����H@��?�33A�Q�B�Q���H@�  >�p�@s33B�                                      By��  �          @�����
@��H?�
=A��B�p����
@�>�33@`��B�p�                                    By�  T          @��R���@��H@{A�
=B�uÿ��@���?(�@�B�W
                                    By�&  T          @�(��Q�@��H@p�A�B�LͿQ�@���?(�@�ffB��f                                    By���  
�          @�����@�(�@�A��HB�#׿��@�G�?\)@�ffB�=q                                    By��r  �          @�녿n{@��\@G�A�33BȽq�n{@�ff>�
=@��
B�\)                                    By��  �          @�z῱�@��\@!G�A�p�B�(����@��?xQ�A\)B��                                    By��  T          @�p���\)@�ff@Q�A���B��`\)@�?L��Ap�B�Q�                                    By��d  �          @����{@�  @Q�A�33B�z΅{@�\)?G�@�=qB��f                                    By�
  T          @��\��p�@�Q�@A�A�(�Bօ��p�@�ff?��HAhz�B�aH                                    By��  "          @�zῧ�@�Q�@1�A�B�aH���@��?�33A4(�B�W
                                    By�)V  �          @�  ����@�33@:�HA�33B�uÿ���@��?�G�AAG�B˔{                                    By�7�  "          @�
=�L��@�p�@1G�A��B�ff�L��@���?�z�A<(�B�ff                                    By�F�  d          @�=q��p�@��?�\)A��B�����p�@���=�?�p�B�=q                                    By�UH  �          @�\)��\)@��?���AUG�B��R��\)@�ff��
=��\)B���                                    By�c�  �          @�����@�{?�  AO\)B�(����@�=q��ff����B�{                                    By�r�  
�          @�{�s33@�\)@�\A�  Bȏ\�s33@�33>�Q�@g
=B�8R                                    By�:  
�          @��W
=@���?��A�p�B��W
=@�33>B�\?���B�                                      By��  "          @�(��^�R@�G�?�p�A���Bƣ׿^�R@�=q=#�
>\BŽq                                    By�  
�          @�=q��33@�=q?�
=AC\)B�Q쿓33@��   ����B��)                                    By�,  2          @�=q�z�H@��?�{A�Q�B�#׿z�H@�\)��\)�333B�=q                                    By��  D          @�녿\(�@��
?�33A7\)B�z�\(�@�ff��R����B�8R                                    By��x  �          @�
=�\)@�G�?��A0��B��{�\)@���(���ָRB�p�                                    By��             @�Q�B�\@��?�p�An�RB���B�\@�������<(�B��)                                    By���  D          @�  =�@��\?���A^�RB��{=�@�\)�����~�RB���                                    By��j  
�          @��>�z�@��?��RAG
=B�
=>�z�@�
=�����
=B�(�                                    By�  
�          @�?}p�@��?(��@�B���?}p�@�\)��z��4��B�Q�                                    By��  
�          @�ff?��\@�ff?�@�(�B��?��\@��H��
=�BffB�33                                    By�"\  �          @��\@z�H@fff�������B*  @z�H@<(��%��ݮBz�                                    By�1  
�          @�Q�@w�@C33������ffB
=@w�@�+���G�A���                                    By�?�  T          @��@�(�@6ff���H���HB
\)@�(�@ff�0  ���A�p�                                    By�NN  �          @��
@��H@8���33��  Bz�@��H@ff�6ff��\A���                                    By�\�  T          @��\@��@p��z�����Aۙ�@��?�\)�>{��G�A��R                                    By�k�  T          @�
=@��?k��"�\����A��@��=��
�,���ͮ?L��                                    By�z@  T          @�G�@��
?����'
=��(�A8��@��
>��
�6ff��
=@J=q                                    By��  
�          @�(�@�?��\�*=q���AA��@�>�p��;����
@fff                                    By�  
�          @���@�\)?�
=�*�H����A333@�\)>�z��9������@0��                                    By�2  �          @���@�G�<�� ������>k�@�G��B�\�����
=C���                                    By��  �          @�@��R?�(��!G���G�Af{@��R?���7
=�Ӆ@��R                                    By��~  �          @�=q@�33?����������A{
=@�33?8Q��4z���z�@��H                                    By��$  �          @�  @�G�?���=q����A|��@�G�?J=q�3�
��
=@��                                    By���  
�          @�=q@��?�{�!���ffA�=q@��?u�?\)���A                                      By��p  
Z          @�33@\?�Q��   ��
=A��@\?�ff�>�R��ffAz�                                    By��  
Z          @�p�@�(�?�  ����=qA7�
@�(�>��#�
����@���                                    By��  
�          @���@�(�?���p����A>=q@�(�?��   ���H@�(�                                    By�b  
�          @�=q@���?����33���AB{@���?���'
=���
@�                                      By�*  �          @���@�  ?�Q��ff��ffA-�@�  >\�'
=���@`��                                    By�8�  
�          @���@�Q�?����ff��{A$(�@�Q�>����%��p�@>�R                                    By�GT  
�          @�33@���?�Q������p�A(z�@���>�G��=q��z�@~{                                    By�U�  
�          @���@��?�p�������A&�H@��>�(��#33���@l��                                    By�d�  �          @�{@׮?���{���Ap�@׮>�p��p���ff@HQ�                                    By�sF  
�          @ٙ�@���?z�H�.{���HA�@���=u�8���ə�?�                                    By��  "          @�ff@�Q�?��������AQ�@�Q�>�\)� �����\@��                                    By�  
�          @��@�?�=q��
���RA��@�>�Q���\��=q@<(�                                    By�8  �          @�{@��H?�\)���H�t��A�
@��H>�(��{����@]p�                                    By��  T          @�@�z�?�Q���m��A
=@�z�?   �������@�G�                                    By�  T          @�33@���?�33����e��A\)@���>��	������@p                                      By��*  �          @�ff@�{?�33���� (�A/�@�{?fff��
=�P(�@���                                    By���  "          @�R@�\)?��ÿ�p��  A%�@�\)?Y���˅�E�@�                                    By��v  T          @�p�@�
=?����=q��AQ�@�
=?����{�A�@��                                    By��  �          @��H@�p�?��
��  �=q@���@�p�?녿�G��2{@�                                      By��  �          @��@��?��\�aG����@�G�@��?.{��z��(�@���                                    By�h  �          @�{@���?�\)�}p���z�A�H@���?=p���ff��R@��                                    By�#  �          @�z�@���?&ff��Q��33@�  @���>aG�����Q�?У�                                    By�1�  T          A ��@�?�Ϳ�����@|(�@�=�Q쿵�!�?#�
                                    By�@Z  �          A�A=q?�R�����@��A=q>u���H�
=?�
=                                    By�O   �          AA��?
=q�}p���=q@n{A��>8Q쿏\)���?��\                                    By�]�  
�          A�A{?녿����ff@x��A{>B�\��Q���?�ff                                    By�lL  T          A�Aff?   �n{���@XQ�Aff>#�
�����=q?�{                                    By�z�  �          A��A�ͽ��Ϳp����ffC��fA�;��ͿY�����C���                                    By�  �          A�RA��  �z�H��p�C�%A�\)�W
=���C�
                                    By��>            AQ�A��k��s33���C�<)A����Q���G�C�7
                                    By���  �          A�A(�<#�
��  ���H=uA(����
�s33��  C���                                    By���  �          A�Az���Ϳfff��ffC���Az���ͿQ����C���                                    By��0  T          A  A\)����Y����Q�C�.A\)���5��C�Z�                                    By���  T          A(�A�����333��ffC���A��\�(��i��C��q                                    By��|  T          A�HA�\=��Ϳ
=q�Mp�?(�A�\���
����N�RC��H                                    By��"  T          A��A��>.{��p��(�?��\A��=#�
������>k�                                    By���  T          A�RA�\=�G���z��33?#�
A�\<#�
���R��\=L��                                    By�n  �          A�
A�
=�G��8Q쿂�\?&ffA�
=L�;W
=��
=>�=q                                    By�  �          A ��A �ͽ#�
�#�
�fffC���A �ͽ�Q�\)�G�C���                                    By�*�  �          A!�A!��L�ͽ��+�C��RA!���Q콸Q��C��q                                    By�9`  "          A!G�A!G��#�
���Ϳ��C�� A!G���\)���
��(�C��=                                    By�H  �          A!�A!��#�
�#�
�W
=C��
A!����
���8Q�C��\                                    By�V�  
�          A!�A!��#�
    <#�
C��A!����=L��>��RC���                                    By�eR  �          A�
A�
��Q�>\)?Q�C��)A�
�#�
>.{?s33C��                                    By�s�  
�          A"{A!녾���>u?�\)C��HA!녾�z�>�Q�@�\C�,�                                    By���  T          A ��A z�Ǯ>�z�?�
=C��A zᾊ=q>��@�C�<)                                    By��D  
�          A#�
A#33�8Q�>�@-p�C���A#33��\?333@y��C���                                    By���  
Z          A#�A"�H�@  ?
=q@A�C��A"�H��?B�\@�Q�C���                                    By���  T          A#�A"�R�@  ?��@EC��HA"�R��?E�@��HC���                                    By��6  
�          A#�A"�\�^�R?:�H@��\C���A"�\��?z�H@��C�aH                                    By���  
�          A"�HA!녿J=q?333@z�HC�A!녿�\?k�@�ffC��                                    By�ڂ  T          A!��A z�fff?B�\@���C�q�A z�
=?��\@���C�S3                                    By��(  �          A%G�A#\)��p�?h��@�G�C��3A#\)�Y��?��\@��C���                                    By���  
�          A$��A#33��{?333@z�HC���A#33�L��?��@���C��H                                    By�t  �          A$��A#
=��p�?E�@�=qC���A#
=�fff?�33@���C�|)                                    By�  T          A%�A"�H���H?Q�@���C��A"�H���?��\@ᙚC��                                    By�#�  "          A$Q�A"{��?u@��HC���A"{��G�?���@�  C�%                                    By�2f  T          A#�A �Ϳ\?�\)@��C��{A �Ϳ��?���AC��                                    By�A  �          A$��A!G���?��
A��C��qA!G��O\)?�A,Q�C���                                    By�O�  
�          A!G�A���H?�p�A�HC���A�\(�?��A,z�C��                                     By�^X  
�          A"{A=q���H?˅A(�C�ФA=q�Tz�?��RA5�C��q                                    By�l�  �          A!A��(�?�33A�\C���A�Q�@�
A<  C��                                     By�{�  
�          A"=qA{���?��A"�HC��A{�0��@
=qAD��C��                                    By��J  "          A$  A���{?�
=A-p�C�  A��(�@�AMG�C�=q                                    By���  "          A"�HA녿�G�@33AP��C�{A녾#�
@   Ac�
C��=                                    By���  
�          A"�HA(��0��@4z�A��\C���A(�>��
@8Q�A���?��                                    By��<  �          A"{A33���@.{AyG�C���A33�u@:�HA��C��{                                    By���  "          A!�A�\�\(�@.{AzffC�w
A�\>�@6ffA�G�?E�                                    By�ӈ  
�          A#\)A�����@3�
A
=C��A�����@FffA�G�C�q                                    By��.  "          A�HA���G�@-p�A}G�C�4{A���  @>�RA���C�>�                                    By���  
�          A ��A
=�У�@?\)A��\C�qA
=���H@W
=A��RC��                                     By��z  
(          A�\A����  @;�A�\)C�s3A���Ǯ@P��A��
C�˅                                    By�   T          A�\A�R����@1�A�=qC��A�R���R@E�A��\C��                                    By��  "          A�\A��
=@:�HA��HC���A����@N�RA�(�C��)                                    By�+l  
�          AffAG���p�@<��A���C�� AG���Q�@Q�A��HC��                                    By�:  �          A�AzῸQ�@Tz�A��C�xRAz�L��@g�A���C�aH                                    By�H�  "          A{A
=��33@R�\A�{C���A
=�.{@dz�A�ffC�u�                                    By�W^  �          A\)A  ��ff@FffA�
=C�1�A  �(�@a�A�\)C��                                    By�f  �          A��A�
��Q�@O\)A�G�C��A�
�0��@mp�A�z�C���                                    By�t�  T          A�A{���R@S33A�C�T{A{�5@r�\A��
C���                                    By��P  
(          A�
A��
�H@QG�A�p�C��qA녿c�
@tz�A�
=C��3                                    By���  
�          A�A���@N{A���C�@ A���G�@s�
A��C�}q                                    By���  �          A�@��R�\)@`  A�G�C�"�@��R�aG�@��A���C��
                                    By��B  T          A��A�
�	��@C�
A�ffC��A�
�p��@g�A���C���                                    By���  �          A�A (��
�H@Z=qA��\C�o\A (��W
=@|��A��HC��                                    By�̎  
�          AQ�A Q���H@hQ�A��
C�  A Q��@��HAׅC���                                    By��4  T          A
=@������@k�A\C�3@���
=q@�(�A�=qC�\                                    By���  T          A��@�Q��@s�
A�G�C��\@�Q�k�@��A�G�C�+�                                    By���  "          A
{@��H���@w�A���C��\@��H=u@��A�p�>�
=                                    By�&  
�          Ap�@��H�h��@u�A��C�z�@��H?   @z=qA�@xQ�                                    By��  �          A��@�  ���@j=qA��
C�n@�  ?��@^{A��HA
�H                                    By�$r  �          @���@��H��@i��A�=qC���@��H?@  @g�A�{@Ǯ                                    By�3  �          @�z�@�G���ff@k�A��
C��@�G�>��R@s�
B ��@>{                                    By�A�  �          @�Q�@�
=���@y��B{C��3@�
=>��@��B
{?�
=                                    By�Pd  T          @�G�@Ǯ�!G�@n{A�33C�)@Ǯ?8Q�@l��A�{@�=q                                    By�_
  �          @�Q�@�  ��=q@p  A�C�Ф@�  ?���@fffA�G�A��                                    By�m�  �          @�33@˅��\)@j�HA�  C��)@˅?��
@b�\A�(�A�                                    By�|V  �          @߮@�\)��ff@e�A��RC��)@�\)?Y��@`��A�33A ��                                    By���  
�          @�R@�
=>�
=@eA��@u�@�
=?�z�@Mp�A�An�\                                    By���  �          @�z�@�  ?!G�@xQ�A��@�  @�  ?��H@Z=qA��A�p�                                    By��H  �          @�=q@�p�>u@u�Bz�@
=@�p�?�=q@_\)A�Ao�                                    By���  
�          @�\@�33>�\)@��
BQ�@5@�33?�(�@p��B�A���                                    By�Ŕ  
�          @�  @�33��@�ffB  C�Ǯ@�33�J=q@�G�B%��C�Ф                                    By��:  �          @�Q�@����N{@���B��C�
=@������R@�p�B7G�C�`                                     By���  "          @�p�@��\���@��B\)C��3@��\�
=q@�  B8�C���                                    By��  T          AQ�@�
=�~�R@�  BQ�C���@�
=��\@љ�BB�HC��                                    By� ,  T          A{@��\�2�\@�{B
=C��q@��\�B�\@��HB8(�C�!H                                    By��  T          @�p�@�Q��p�@�{B ��C�q@�Q콸Q�@�=qB0  C���                                    By�x  �          @��R@�\)��@��B&�
C��H@�\)>�  @��
B1�@(��                                    By�,  T          @�@���@�p�B"�C�n@���@��RB9p�C�"�                                    By�:�  "          @�33@�33�'
=@�\)B$  C�t{@�33�(��@��\B=ffC�L�                                    By�Ij  �          @�@�{�O\)@�33B!��C��\@�{��ff@�{BE�
C�G�                                    By�X  �          @��@�ff��H@��B5�RC�^�@�ff��33@�(�BM�C��                                    By�f�  �          @�  @g��6ff@��BO\)C���@g���@ϮBr=qC��\                                    By�u\  �          @��
@�ff�B�\@�=qB-  C��\@�ff���
@�=qBO��C�z�                                    By��  "          @��@�p��c33@�\)B��C�g�@�p���\)@�{BD  C�s3                                    By���  �          A�H@�Q��\)@�\)B"(�C��)@�Q��\@�G�BK�C�J=                                    By��N  T          A�@�Q��j�H@�33B({C���@�Q쿵@�G�BM�
C��                                    By���  �          A@�
=�g�@�  B*�C�E@�
=��z�@�BR
=C��\                                    By���  �          @��
@�\)�^�R@�{B/ffC��@�\)��ff@ʏ\BV�\C���                                    By��@  �          @���@��R�N{@�(�B5��C�S3@��R��=q@�p�B[G�C��R                                    By���  �          @�@����2�\@�\)B@C���@��ÿ#�
@��
Ba�C���                                    By��  
�          @��H@����>�R@��B;��C���@��ÿ\(�@�=qB_ffC���                                    By��2  �          @��@����w
=@�=qB�\C�  @������@�{BB�RC��                                    By��  "          @�@�  ��ff@}p�B =qC��f@�  �.�R@���B4��C�`                                     By�~  
�          @�=q@y�����@W�A�z�C���@y���QG�@��
B.��C���                                    By�%$  	`          @�G�@w����\@i��A��RC�R@w��=p�@���B833C�J=                                    By�3�  T          @�
=@s33���\@C�
A�ffC��)@s33�w�@���B%\)C�8R                                    By�Bp  
�          @��@l(���\)@QG�Aޣ�C�E@l(��^{@�33B.�C�c�                                    By�Q  �          @�Q�@Z=q���H@@  A���C�N@Z=q�y��@�
=B)ffC��)                                    By�_�  T          @�33@aG���z�@Y��A�z�C��{@aG��Fff@��HB7C�K�                                    By�nb  "          @ָR@�(��\��@]p�A�p�C���@�(���(�@�
=B&z�C�y�                                    By�}  T          @�@�p��Fff@q�B��C�8R@�p����
@�z�B-z�C���                                    By���  
�          @�ff@������
@���Bp�C�H@������@��B;�RC�]q                                    By��T  T          @���@�z��h��@��RB33C�*=@�z���
@�\)BD�\C���                                    By���  �          @�G�@�33�E�@��B,Q�C�` @�33����@�Q�BPp�C�)                                   By���  T          @�ff@�G��AG�@�B,�C�n@�G����
@�BP��C�:�                                   By��F  �          @�  @����0��@�(�B4��C���@����8Q�@�G�BT�C�AH                                    By���  "          @�=q@�Q���R@��B>�\C��=@�Q쾸Q�@�{BX�C���                                    By��  �          @߮@�p��(�@��B=\)C�#�@�p��B�\@���BTffC���                                    By��8  �          @�(�@�  �p�@�B3p�C�7
@�  ���@�  BQp�C���                                    By� �  
�          @��
@�  �"�\@��
B1  C��\@�  �0��@�\)BPz�C�{                                    By��  
�          @�33@|(��!�@�z�B2�C��H@|(��.{@�  BR�C��                                    By�*  
�          @�@�  �%@�p�B1C���@�  �8Q�@���BQ�C��                                    By�,�  �          @�@z=q�.�R@�p�B1��C��@z=q�Y��@�33BT��C��                                     By�;v  "          @�z�@o\)�8Q�@�B2��C�<)@o\)�z�H@���BY�C���                                    By�J  T          @�ff@c33��G�@C�
AمC�AH@c33�Vff@��HB-  C�K�                                    By�X�  	�          @�33@W���
=@.�RA��
C��@W��h��@�33B%
=C�ff                                    By�gh  T          @��H@5�1�@�BL\)C��3@5�L��@�33Bx�RC�                                      By�v  �          @�33@/\)�s33@���B3{C��=@/\)����@��RBo��C��)                                    By���  
�          @Ӆ@8Q��tz�@���B.G�C�|)@8Q��z�@�(�Bi�C�8R                                    By��Z  "          @�ff@\)�w
=@���B2�C�ff@\)���H@��BrQ�C��                                    By��   "          @��@(Q��w�@�(�B,z�C�!H@(Q�� ��@��Bkz�C�S3                                    By���  
�          @���@>{�l��@��HB#
=C�c�@>{���H@���B]z�C�L�                                    By��L  �          @���@AG��X��@��\B.�\C��\@AG��˅@���BdffC��                                    By���  T          @��H@G
=�X��@��HB-=qC�L�@G
=��=q@���Ba��C���                                    By�ܘ  
(          @���@E�tz�@�33BQ�C�xR@E�z�@��RBY�C�
                                    By��>  �          @��
@A���=q@vffBC�Z�@A����@�=qBRC��                                    By���  T          @�G�@.�R��@a�B�C��
@.�R�5@���BL��C���                                    By��  �          @�ff@Q����R@P  A���C�j=@Q��Mp�@��BG��C�P�                                    By�0  "          @��
@{���@>{A���C��f@{�U�@��B==qC�J=                                    By�%�  T          @���@$z���(�@/\)A�33C�� @$z��c33@�=qB2��C���                                    By�4|  "          @�\)@4z����R@\)A���C���@4z��mp�@�(�B&�C��                                     By�C"  
�          @��@S�
�n{@=p�A��C�޸@S�
�Q�@��B4�C�%                                    By�Q�  �          @��H@Q��e�@P  B=qC�Ff@Q��	��@��\B?\)C�j=                                    By�`n  S          @�G�@P  ��Q�@$z�A��C��f@P  �333@w
=B'�HC��=                                    By�o  �          @��@Fff���@p�A��C���@Fff�AG�@c33Bp�C��q                                    By�}�  T          @�
=@J=q���@�\A�33C�� @J=q�B�\@i��B  C�3                                    By��`  �          @��H@Q����@�A�p�C���@Q��N{@fffBffC��f                                    By��  �          @��\@Tz���Q�@
=qA�Q�C��@Tz��K�@dz�B=qC�R                                    By���  �          @��@Z�H���
@  A���C��R@Z�H�@��@fffB\)C�N                                    By��R  "          @���@8Q��B�\@g
=B#(�C��{@8Q��G�@�
=BW��C�                                      By���  �          @�  @Q��33@�=qBc��C���@Q���@�\)B���C�@                                     By�՞  �          @�\)@"�\��R@��
BW  C�Z�@"�\�Ǯ@��
B}�C��
                                    By��D  	�          @�{@'
=�%@��HBF��C���@'
=�Tz�@��Bt�
C�/\                                    By���  �          @�z�@-p��<��@}p�B2�C�L�@-p���ff@�Q�Bf��C�33                                    By��  �          @�\)@Fff���ý��
�=p�C��\@Fff���R?�\A��C���                                    By�6  �          @�G�@L�����?�\A�=qC���@L���u�@Y��B	�C��\                                    By��  �          @���@Vff��=q?�=qAK�C�ff@Vff��  @?\)A�RC��
                                    By�-�  �          @���@O\)��z�?�{AQ��C��=@O\)����@C33A��C�XR                                    By�<(  �          @�33@qG���  ?
=q@��
C�=q@qG���
=@\)A���C��f                                    By�J�  �          @��@e���  ?z�@���C�� @e����R@G�A���C�.                                    By�Yt  �          @�ff@Vff����?�\)A-C���@Vff��G�@2�\A�C�ٚ                                    By�h  �          @��
@<(���
=?��
A�  C��\@<(��fff@C�
BffC���                                    By�v�  �          @�G�@G��c�
@��A�G�C�@ @G��!�@R�\B0p�C���                                    By��f  �          @�\)@��j=q@(Q�A��C��@����@r�\BF��C�.                                    By��  �          @�(�@\)�tz�@$z�A�
=C�33@\)�'�@r�\B>�
C�H�                                    By���  �          @�@'��n{?�\)A�ffC���@'��1�@FffB
=C���                                    By��X  �          @��H@4z��j�H?�
=A�\)C���@4z��-p�@H��B
=C�
                                    By���  �          @��\?�p��`��@L(�B��C�  ?�p��ff@�Q�Be�C��)                                    By�Τ  �          @�33?�=q�e@G
=B�
C���?�=q���@��RB_{C���                                    By��J  �          @�=q@���Q�@.{A�=qC�AH@��/\)@�  BC{C�>�                                    By���  �          @��@(��{�@333A���C��\@(��(��@�G�BF�RC�ٚ                                    By���  �          @�G�@Q��i��@>{B�C��=@Q��z�@�33BK�
C��H                                    By�	<  �          @���?�Q��P��@]p�B)\)C�]q?�Q���
@��Bl�
C��f                                    By��  �          @���?���R�\@c33B-�C�Ff?����\@�Q�Bs(�C���                                    By�&�  �          @��?�=q�\(�@N�RB�\C�?�=q�G�@�Q�Bez�C��                                    By�5.  �          @���?���]p�@G�B\)C�N?����@��B_��C�
                                    By�C�  �          @��R?�33�`��@E�B��C��3?�33�	��@���Bcp�C��                                    By�Rz  �          @�Q�?�{�W�@U�B&C�˅?�{��@��\Bo�\C�H                                    By�a   �          @�\)?�{�Tz�@U�B(G�C��?�{���@��BpC�P�                                    By�o�  �          @��?�z��@��@mp�B<��C�b�?�zΌ��@���B�\)C�\)                                    By�~l  �          @�?���@  @~�RBI
=C�k�?�녿��@���B�#�C��q                                    By��  �          @�(�?��7
=@�33BT�C��?���z�@��B��C��                                    By���  �          @�33?��1G�@��
BW��C�xR?�����@�33B��RC��H                                    By��^  �          @��?�{�1G�@tz�BP{C���?�{��
=@��B��)C���                                    By��  �          @�{?L����@�z�B}p�C��?L�;�{@��B���C��                                     By�Ǫ  �          @�Q�?xQ��?\)@u�BJ=qC�H?xQ쿱�@��B��C�}q                                    By��P  �          @��H?c�
�
�H@�\)Bs�HC��?c�
��@�\)B��fC���                                    By���  �          @���?G��   @���B�\C��q?G����@�=qB�C�޸                                    By��  �          @�p�?fff��\)@�G�B���C���?fff>��@��B��\A�\                                    By�B  �          @���?E�����@�Q�B�z�C�*=?E�>��@�{B�ǮA�R                                    By��  �          @��\?p�׿��@��B��RC�f?p��>���@���B�=qA���                                    By��  �          @�p���z�?�33@�Q�B�#�B�L;�z�@!�@S�
BQ��B�
=                                    By�.4  �          @�����
�.{@��HB���CuE���
?�@��B�ǮB�k�                                    By�<�  �          @�=q��Q����@��B��=Cdc׽�Q�?���@�G�B���B��3                                    By�K�  �          @���    ��\)@���B�Q�C��    ?˅@�Q�B���B�W
                                    By�Z&  �          @���Q��@��B�.Ciٚ��Q�?�=q@�z�B�Q�B���                                    By�h�  �          @��H>�׾Ǯ@�\)B�z�C��>��?�G�@��B�G�B�k�                                    By�wr  �          @��?녿
=q@�G�B�.C�  ?�?�ff@�{B�#�Bvp�                                    By��  �          @��R?�\���@�z�B��=C�y�?�\?�  @���B�#�B{�R                                    By���  �          @�?!G��@  @�=qB�p�C�ٚ?!G�?^�R@�G�B�BX�                                    By��d  �          @�(�?��B�\@���B���C�W
?�?Y��@�Q�B���Bh�                                    By��
  �          @�(�?(�ÿ}p�@�ffB�z�C���?(��?(�@���B�\)B+                                      By���  �          @���?0�׿�{@�=qB��C��?0��>�ff@�{B��HBp�                                    By��V  �          @�?(�ÿ�ff@��B�8RC��?(��?\)@��\B�\)B�R                                    By���  �          @���?fff�z�@�z�B�u�C���?fff?�ff@���B�Q�BE(�                                    By��  �          @�G�?��O\)@��B�L�C��?�?fff@���B�  B{                                    By��H  �          @�G�?�\)�p��@�\)B�=qC�Ǯ?�\)?@  @���B���A���                                    By 	�  �          @��H?��^�R@���B���C�=q?�?Tz�@�G�B��A�\                                    By �  �          @�(�?��>B�\@��B��A
=?��?��@�33B|G�Bl33                                    By ':  
�          @���?:�H?�@��B���Bhz�?:�H@*�H@h��BRB�u�                                    By 5�  �          @��?�R@G�@_\)Bk
=B���?�R@I��@!G�Bz�B��\                                    By D�  �          @{�?��@�
@G
=BR  B��H?��@Q�@�
A��B�B�                                    By S,  �          @��H?333@8Q�@6ffB/Q�B��R?333@n{?�{A�z�B��                                    By a�  �          @y��?z�@:=q@p�B�\B�\)?z�@g
=?��RA�  B���                                    By px  �          @tz�>.{@(�@G�B[z�B��)>.{@J=q@
=B��B��                                    By   �          @xQ�?+�@��@FffBS(�B�33?+�@N�R@�B B��R                                    By ��  �          @p��?@  @��@@  BSp�B�B�?@  @E�@G�B  B��                                    By �j  �          @l(�>��
@
=@@��BZQ�B��\>��
@C�
@�\BG�B��f                                    By �  �          @r�\=���?�@P��Bn�B��=���@>{@ffB�B�33                                    By ��  �          @qG�>��?�ff@S�
BuQ�B��=>��@8Q�@�B �
B�\                                    By �\  �          @��>�  ?�p�@aG�Bq�
B��=>�  @G�@$z�B��B�Ǯ                                    By �  �          @�G�>W
=?�33@vffB~�B���>W
=@J=q@9��B*=qB��\                                    By �  �          @�(�>��@�@�B�ǮB�Ǯ>��@_\)@Y��B1(�B�8R                                    By �N  �          @��H>�?�=q@�\)B�p�B��)>�@U�@aG�B:�B��                                    By�  �          @�z�=u?��H@��B��qB���=u@I��@Z=qB={B��q                                    By�  �          @�����
?���@���B�\B��ü��
@7�@g
=BM�RB��                                    By @  �          @��>W
=?�ff@���B���B�� >W
=@?\)@Z�HBCG�B��H                                    By.�  �          @��
?�?��@�\)B�u�B�Ǯ?�@R�\@b�\B:B�W
                                    By=�  �          @���?��\@z�@�Bu(�Bi?��\@aG�@X��B(�B�W
                                    ByL2  �          @�{?��?���@�=qBt=qBZ�?��@W�@U�B)�
B�B�                                    ByZ�  �          @��\?�(�?�
=@�ffBu��BS�?�(�@Y��@]p�B,(�B�Q�                                    Byi~  �          @���?���?��@��B}{BQQ�?���@Q�@c33B3��B�Q�                                    Byx$  �          @��?��?�(�@�33B~z�BS=q?��@J�H@\��B4�B�33                                    By��  �          @�
=?�Q�?�
=@�{B~{BEp�?�Q�@J=q@c33B6\)B���                                    By�p  �          @�  ?�
=?�Q�@�\)B~�BF?�
=@K�@e�B6B�z�                                    By�  �          @�=q?��H?�=q@��HB���B<?��H@G
=@n{B<��B�Ǯ                                    By��  �          @���?��\?�
=@�(�B}�By�?��\@W�@Y��B/��B�u�                                    By�b  �          @�Q�?z�@��@�ffB{�B�k�?z�@h��@W�B)B��                                    By�  �          @��>�33@\)@�Q�Bsz�B���>�33@{�@Tz�B =qB��
                                    Byޮ  �          @�p�>Ǯ@,��@���Bh�B�u�>Ǯ@��H@I��B��B�\                                    By�T  �          @��H>�33@/\)@�  B]p�B�B�>�33@\)@0��B
=qB��
                                    By��  �          @��
?:�H?�
=@|(�B�(�B�(�?:�H@>�R@E�B4�RB��                                    By
�  �          @�{?#�
?�  @���B�u�B��?#�
@Dz�@I��B4\)B�L�                                    ByF  �          @�G�?!G�?�\@vffB�  B���?!G�@AG�@>{B/ffB�\)                                    By'�  �          @�?333?��H@z�HBx�B��{?333@N�R@>{B'�B�u�                                    By6�  �          @�
=?��@
�H@���Bv��B��?��@`  @G
=B$�
B�#�                                    ByE8  �          @�
=>\?�
=@�(�B��B�z�>\@HQ�@_\)B?�B��                                    ByS�  �          @�
=>�  ?�Q�@�
=B���B�\)>�  @6ff@\(�BI
=B�                                      Byb�  T          @��=#�
?�
=@�  B��fB��{=#�
@'
=@c33BV��B�u�                                    Byq*  �          @��R?�?�  @�\)B��B�=q?�@*�H@`��BQ{B��                                    By�  �          @�=q>�33?0��@��B�Q�B{��>�33@	��@n{Bn�RB�\)                                    By�v  �          @�G�>\>�@�
=B��HBL�>\?�
=@q�Bz�B���                                    By�  �          @�  >�\)>aG�@��RB��fB\)>�\)?��
@��HB�k�B��                                    By��  �          @�z�=�    @�(�B��C�
=�?У�@��\B��\B��)                                    By�h  �          @�33>���<�@��\B�Ǯ@�=q>���?�33@���B�{B��                                    By�  �          @���=���@���B��C�ٚ=�?���@��B�G�B�p�                                    By״  �          @��>\>�@�
=B�ǮA�=q>\?�Q�@�z�B�u�B�                                    By�Z  �          @���>�녾\)@���B�{C�j=>��?��H@�G�B���B���                                    By�   T          @��>B�\�\@�G�B�B�C�g�>B�\?��R@�(�B���B�aH                                    By�  �          @�  =L�Ϳ��
@��B���C���=L��?(��@�B��)B��{                                    ByL  �          @��ͽu��Q�@�(�B�ǮC���u>�@��B��3B噚                                    By �  �          @�p���G��޸R@��
B�L�C�B���G�>.{@���B���B��H                                    By/�  �          @��R�.{��
=@��B��C�
�.{=u@�ffB�ǮC 5�                                    By>>  �          @���>��
=@�\)B�33C���>���@��B�  C��\                                    ByL�  �          @��=�\)�\)@�Q�B��RC��
=�\)��{@�\)B���C�g�                                    By[�  �          @���>W
=���@��\B��C�� >W
=�k�@�  B���C�4{                                    Byj0  �          @���>�� ��@��
B�C���>���Q�@�  B���C���                                    Byx�  �          @��?!G����@��HB��\C�Ф?!G�����@���B�\)C��                                    By�|  �          @�Q�?:�H� ��@��B�=qC��
?:�H��G�@�ffB���C��)                                    By�"  �          @�\)?fff����@��\B���C��R?fff=#�
@��B�
=@G�                                    By��  �          @��\?����Q�@�{B�aHC���?��    @�G�B���C��H                                    By�n  3          @���?
=q��\)@�B��)C��?
=q=L��@�  B��R@���                                    By�            @��>�p��   @��HB�p�C�1�>�p�����@�
=B�
=C�'�                                    Byк  �          @�\)>�\)��{@���B��)C�Ff>�\)=L��@�
=B�{AQ�                                    By�`  �          @�{>\)����@���B��qC��
>\)?��@��B��B�                                    By�  �          @��׼#�
��Q�@��\B��RC����#�
?   @�  B���B���                                    By��  �          @�\)������@�Q�B��)C��{���>�@��B��HB�u�                                    ByR  �          @�ff�L�;�(�@�B�\C��R�L��?��@���B�L�B�p�                                    By�  �          @�  =u�B�\@��B��C�S3=u?�ff@�  B��RB�ff                                    By(�  �          @��
�Ǯ���@��B���C�=�Ǯ>u@��HB�C�=                                    By7D  �          @�
=���H���@�ffB�
=C|(����H>8Q�@�{B���C                                       ByE�  �          @����\���@�ffB�
=Cxn��\>��@��B���CY�                                    ByT�  �          @����(�����@���B�33C|xR��(�>�z�@��RB�z�C��                                    Byc6  �          @����Q쿵@��
B�  C녾�Q�>.{@��HB�#�C�                                    Byq�  
�          @��ͿaG��#�
@��B�B�CX+��aG�?\(�@�ffB�u�C��                                    By��  �          @�
=�:�H��p�@�\)B��=Co� �:�H>�z�@�z�B�C\                                    By�(  �          @�녿c�
��Q�@�G�B��3CiJ=�c�
>�=q@��RB�#�C"�R                                    By��  �          @�{�L�Ϳ�  @\)B�aHCg���L��>�(�@��HB�{C�f                                    By�t  �          @���L�ͿxQ�@���B���Cf���L��>�@�z�B��C�H                                    By�  �          @��H�s33���@��\B�
=Cf0��s33>���@�
=B�B�C!�                                    By��  �          @��\��\)����@�Q�B�u�Cc+���\)>u@�B�p�C'�f                                    By�f  �          @�p����
�s33@��
B�ǮCX�����
?�@�ffB���C��                                    By�  �          @��\�������@��B��3CEc׿��?��\@�  B�W
Cz�                                    By��  �          @��ÿ+��.{@�33B���CaT{�+�?B�\@��HB�(�C�                                     ByX  �          @|(��fff��G�@r�\B�L�CN@ �fff?^�R@n{B��3C�                                    By�  �          @x�ÿ����@l��B�W
CPǮ���?8Q�@j�HB���Ck�                                    By!�  �          @tzῗ
=��G�@fffB�G�CHh���
=?O\)@a�B��
C�q                                    By0J  �          @|(���ff�c�
@l(�B�(�C\LͿ�ff>�
=@qG�B��qCY�                                    By>�  �          @xQ�p�׾�33@p  B�  CHk��p��?p��@h��B��fC�                                    ByM�  �          @u�\)��Q�@Z�HB�
=Ct�\)=#�
@g�B�aHC/��                                    By\<  �          @w
=��=q���
@Z=qBx��C��)��=q��@s�
B�aHCrB�                                    Byj�  �          @qG��\)����@W
=B~\)CzW
�\)��{@l(�B�CS�\                                    Byy�  �          @g��\)��@<(�BY�\C~�f�\)�k�@\��B�aHCn��                                    By�.  �          @�  ��  ��z�@fffB��\Cj���  ��@w�B��qC:�{                                    By��  �          @�����
��ff@p��B�\)Cg�=���
=#�
@~{B��HC1�                                    By�z  �          @�����ÿ�@hQ�Bu��Cms3���þ�33@~�RB��CFJ=                                    By�   �          @��Ϳ�  ���R@Z�HB^(�Cm���  �:�H@x��B�L�CR=q                                    By��  �          @�=q���\����@]p�BjQ�Cq.���\�
=@xQ�B�Q�CQ�H                                    By�l  �          @s�
�\)��z�@Mp�Bl�C����\)�:�H@j�HB�{C���                                    By�  T          @y��?@  �@L��B]  C��3?@  �fff@n{B�
=C���                                    By�  
�          @x��>��
�{@J=qBZ��C�R>��
���@n{B�� C���                                    By�^  �          @�=q�u��@Q�BX=qC�K��u��33@xQ�B�
=C���                                    By  �          @w
=�L���z�@Dz�BS�C�~��L�Ϳ�@j=qB�(�C�\                                    By�  �          @|(���=q�=q@FffBO��C����=q��  @n{B�.C��\                                    By)P  �          @��R���
���@Z=qB[=qC����
���@�Q�B�ffC�}q                                    By7�  �          @��
��Q��0��@VffBIG�C�B���Q쿾�R@��\B�ǮC�+�                                    ByF�  �          @�z���$z�@N�RBKffC�H�����{@y��B�  Cy\                                    ByUB  �          @l�;�p����H@UB�\)C� ��p���\)@hQ�B�z�CX��                                    Byc�  �          @g
=��z��-p�@�B#��C��;�z���
@HQ�BpG�C�T{                                    Byr�  �          @b�\��\)�{@\)B3�C��ᾏ\)��G�@J�HB�{C���                                    By�4  �          @��\<��
��Q�?�Q�A�{C�'�<��
�S�
@+�B��C�0�                                    By��  �          @�z�>�\)���׾�33���RC��f>�\)��(�?�33Ak33C��{                                    By��  T          @��?   ��
=>�@���C�,�?   ��=q?�AȸRC�|)                                    By�&  �          @��H>�����=�\)?h��C���>����G�?ǮA�C��)                                    By��  �          @��u���
?�RA ��C��ýu�{�@G�A�\)C��=                                    By�r  �          @�ff�W
=���>��?�\C���W
=���H?�  A���C��R                                    By�  �          @��H�.{��G��   ���
C���.{��p�?�
=AV�HC��                                    By�  �          @��>�(����<#�
=uC�^�>�(���(�?�Q�A�(�C���                                    By�d  �          @�Q�#�
��  =�Q�?z�HC�  �#�
��{?��A��C��                                    By
  �          @��׾�����  >8Q�@ ��C����������?��A�G�C��f                                    By�  �          @�
=    ��ff�8Q���C�H    ��\)?�G�A�G�C�H                                    By"V  �          @�p���������>8Q�?�(�C�N�������?�=qA��RC�/\                                    By0�  �          @�Q쾀  ��  =�\)?@  C����  ���R?�
=A���C�w
                                    By?�  �          @����{������������C�\��{���?�  Adz�C�H                                    ByNH  T          @�>�������
�e�C���>����?�{As�C��                                     By\�  �          @�{>W
=���Ϳ�����C�*=>W
=����?�33AM�C�0�                                    Byk�  �          @��<��
��33�0����Q�C�"�<��
����?z�HA0  C�"�                                    Byz:  �          @��?���33�\(�� Q�C��?����
?=p�A	��C��                                    By��  �          @�G�?���������]p�C�%?����>�
=@�  C�\                                    By��  �          @�  ?
=q��G���=q�w
=C�5�?
=q���R>�=q@G
=C�R                                    By�,  �          @���>B�\��Q��=q���RC�*=>B�\����<��
>�\)C��                                    By��  �          @�33=�Q����\��{���C��=�Q����H<��
>W
=C��f                                    By�x  �          @�=q����{��
=��z�C�˅����녾����o\)C��\                                    By�  �          @�{>L�����R�
�H��C�9�>L�����Ϳ\)��ffC��                                    By��  �          @�ff>#�
��p��z���
=C���>#�
��p��8Q���(�C�޸                                    By�j  �          @��R>k���G���Q��T��C�O\>k����>��@���C�G�                                    By�  �          @�{�#�
����� �����HC����#�
��������I��C�*=                                    By�  �          @�{������Ϳ�Q����HC}q�����G���ff��\)C�O\                                    By\  �          @�  ��(�����������C���(�������
�H��C�O\                                    By*  �          @��
�z���������݅C�(��z����׿L����
C��                                    By8�  �          @��þ����{��*=q���C���������=q����{�C��3                                    ByGN  �          @��R>�z��^�R�.�R�{C�\)>�z���p���  ��Q�C���                                    ByU�  �          @}p�<��#�
�>�R�E33C�O\<��W
=�G���ffC�<)                                    Byd�  �          @r�\�������C�
�X�
C������C33�{�{C��{                                    Bys@  �          @hQ콏\)���9���T33C�ͽ�\)�>�R����C�N                                    By��  �          @�  ��\)��\�P���Z�RC����\)�L�������C�~�                                    By��  �          @s�
�������QG��lC��������5� ���$�HC���                                    By�2  �          @g���{��
=�J�H�v�RC�=q��{�%��   �/Q�C�9�                                    By��  T          @g
=�.{����Tz�Cn!H�.{��
�4z��Q�\C{�
                                    By�~  �          @aG�������Q��R�\�C�{�����	���0���O\)C�                                    By�$  T          @]p����Ϳ���P���{Cy�
�����G��1G��U�C�s3                                    By��  �          @y��?\)��R��R�1G�C�j=?\)�G��������
C�                                      By�p  �          @�G�?��
��G���������C��
?��
����\��=qC�>�                                    By�  �          @�{?�\)��(��dz���C��)?�\)�����33���\C���                                    By	�  �          @��H?�����{�l(��p�C�+�?������H�   ����C���                                    By	b  �          @�  ?��������w
=�p�C�3?�����
=�p���  C��\                                    By	#  �          @�(�?�=q��(��xQ��C�� ?�=q���\������C�n                                    By	1�  �          @�=q?��
��{�����+�C��?��
��G��/\)���C���                                    By	@T  �          @ʏ\?�\)��33���
�/(�C��
?�\)��\)�5���C�H                                    By	N�  T          @�33?�  ����<(���RC�g�?�  ��zῚ�H�733C��3                                    By	]�  �          @�p�?����z��X����C�b�?����{��  ��  C�b�                                    By	lF  T          @�Q�?ٙ���z���(��$��C���?ٙ���{�'
=����C��3                                    By	z�  �          @���?�33��=q�~�R�Q�C��=?�33�����H��
=C�B�                                    By	��  T          @ə�?�(����{����C��{?�(���������RC�H�                                    By	�8  �          @�{?����{�^{�Q�C��?����  ��=q��G�C�Ф                                    By	��  �          @���?�=q������R��=qC��\?�=q��=q�8Q��z�C�G�                                    By	��  �          @Ǯ?�z������L������C�q?�z���Q����g
=C�f                                    By	�*  �          @�p�?�Q���=q�����$ffC���?�Q����\�$z��ř�C�Ǯ                                    By	��  �          @���?�����33����2��C�� ?�����ff�8����p�C���                                    By	�v  T          @��?��H�����33�3�RC��?��H��
=�;���G�C��\                                    By	�  �          @Ǯ?���������33�0��C���?�����(��8Q���  C��                                    By	��  �          @��?�  ��z������*��C�o\?�  ���R�1���G�C��\                                    By
h  �          @�Q�?\������  �G�C�<)?\������R���HC��\                                    By
  �          @�\)?W
=�������Q�C�<)?W
=���
�(�����C��=                                    By
*�  �          @���?^�R��\)�Q���p�C�'�?^�R���H��\)��RC��                                    By
9Z  3          @�33?u��{��\���C��)?u�ȣ׾L�Ϳ�C�]q                                    By
H   u          @�  ?�(���G��   ����C���?�(����
�aG��   C��                                    By
V�  �          @�G�?�z��������(�C��
?�z����Ϳ   ���\C�U�                                    By
eL  �          @�p�?�\)���
�ff��{C�g�?�\)�ə��
=q��=qC�
=                                    By
s�  �          @�ff?��
��G��*�H���C��?��
��=q�^�R��ffC��                                     By
��  �          @�{?u��
=�3�
��z�C��q?u��G����\�{C�Q�                                    By
�>  �          @�{?�����R�0����
=C�C�?���ȣ׿z�H�z�C��\                                    By
��  �          @�
=?aG�����6ff��33C�XR?aG���=q�����(�C��3                                    By
��  �          @�  ?^�R����2�\����C�<)?^�R���
�}p��(�C�޸                                    By
�0  �          @�ff?fff��  �0����
=C�n?fff�ə��}p��G�C��                                    By
��  �          @�z�>������\)��C�\)>���  �8Q���=qC�.                                    By
�|  �          @�(�?�{�aG��\(��)��C���?�{�����\��G�C��f                                    By
�"  �          @�(�?˅����r�\�[��C�Y�?˅�HQ��AG��"�
C�y�                                    By
��  �          @�\)?�녿��������jz�C��?���8���U��4G�C��3                                    Byn  �          @���@   ��
=�{��[33C�W
@   �  �[��5��C�H                                    By  �          @��@�R��  �{��hp�C��R@�R�z��_\)�C\)C���                                    By#�  �          @�@#�
�L���x���b=qC�@#�
����j�H�P  C��
                                    By2`  �          @��H@%?+��i���V��Afff@%�Ǯ�l(��Z\)C���                                    ByA  �          @��H@z�>���e�b�A6=q@z���e��a(�C�'�                                    ByO�  �          @z�H?��R���Tz��i=qC�*=?��R��33�C33�M��C�u�                                    By^R  �          @��H@����Y���b�C�.@����H�G
=�GQ�C�
                                    Byl�  �          @���@{�}p��aG��]��C���@{��Q��G
=�:  C�j=                                    By{�  �          @��H@G���
=�^�R�V�RC�L�@G��ff�@���0�HC���                                    By�D  �          @���?�ff� ���B�\�@�C�P�?�ff�P  �{��
C��                                    By��  u          @�G�=�G��s�
�p��p�C�ٚ=�G����H�����C���                                    By��  �          @�33?
=q�j=q�.{�=qC�4{?
=q���׿�������C��)                                    By�6  �          @�  ?���`���&ff�G�C���?�����H���
����C�w
                                    By��  �          @��?��H�[��=q�  C��q?��H�}p���\)��(�C���                                    Byӂ  �          @���?��\�u���ڣ�C�1�?��\��Q�s33�AG�C�T{                                    By�(  �          @��?����g
=����G�C�z�?�����  �Y���9p�C��3                                    By��  T          @�z�?���\(��G���ffC���?���w
=��  �`Q�C��                                    By�t  �          @���?��e��ff��ffC�z�?��xQ��\��C�'�                                    By  �          @^{�p���S33>�@�ffC~)�p���C33?���A�
=C|޸                                    By�  �          @e�Ǯ�c�
>L��@HQ�C��H�Ǯ�W�?�z�A�z�C��{                                    By+f  �          @g��#�
�a�>��H@�33C�޸�#�
�P��?�
=A��\C�u�                                    By:  �          @l�Ϳ�\�e�?E�A@z�C����\�O\)?��HA�ffC���                                    ByH�  �          @i����(��X��?��RA��C�W
��(��;�@
=B  C��f                                    ByWX  
�          @n{�=p��[�?�A�\)C��f�=p��?\)@33B�C��                                    Bye�  u          @e��{�Z=q?��A�p�C�&f��{�@  ?�B�C��                                    Byt�  �          @g��
=q�Vff?�(�A�z�C�k��
=q�9��@z�Bp�C���                                    By�J  �          @n{�.{�]p�?�
=A��\C�o\�.{�AG�@�
B�C��                                    By��  �          @o\)�(��_\)?�Q�A�C���(��C33@z�B��C�XR                                    By��  �          @���W
=�u�?�\)A��
C�Ф�W
=�U�@�B�C�
                                    By�<  �          @�{�h���w
=?��A��HC�aH�h���W�@�
B
=C~��                                    By��  �          @��R�J=q�z�H?��A�\)C�Ff�J=q�\(�@33B��C�}q                                    By̈  �          @�ff�Q��{�?�G�A��C�  �Q��]p�@  Bp�C�Y�                                    By�.  �          @����  �vff?�(�A��Cuÿ�  �Y��@(�A�p�C}��                                    By��  �          @��u�z=q?�\)AyC�ÿu�^{@
=A�\C~�=                                    By�z  �          @��H��z��qG�?�=qAt��C|녿�z��W
=@�A홚Cz��                                    By   �          @�Q쿣�
�j�H?\(�AG�Cz�q���
�Tz�?��A�\)Cx�H                                    By�  3          @�(�����c33?У�A��
Cy^�����@  @\)BCv                                    By$l  u          @�p���  �e�@{A��Cz�ῠ  �8Q�@E�B1��Cv�                                    By3  �          @��Ϳ����X��@%�B�C|
=�����&ff@W�BG�Cw&f                                    ByA�  �          @�z῎{�Y��@"�\B��C{�῎{�(Q�@U�BE�RCw0�                                    ByP^  �          @�\)�\�C�
@(�B
=Cs�)�\��@H��BA��Cl��                                    By_  �          @�Q쿚�H�XQ�@B�CzJ=���H�*�H@HQ�B;G�Cu��                                    Bym�  �          @}p����R����@A�BH�\CZT{���R�=p�@UBf(�CH�                                    By|P  �          @z=q�	���\@O\)B`(�C=��	��>�@N�RB_  C'�H                                    By��  �          @���z���\@>{B;  Cb�{��z῜(�@Y��Ba  CTxR                                    By��  �          @|�Ϳ�Q��&ff@&ffB$�RCp�q��Q����@J�HBU��Ch(�                                    By�B  �          @|�Ϳ\��@.�RB.�RCm�q�\��z�@P��B]��Cc��                                    By��  �          @}p����R���@.{B-�\Cn�����R��Q�@P  B\�
Cd�=                                    ByŎ  T          @}p�����'�@'�B&
=Cq�q��녿�\)@L��BWp�Ciff                                    By�4  T          @z�H��  �.{@��Bp�Cq���  ��@@  BGffCiz�                                    By��  �          @|(���  �33@?\)BC33Cqzῠ  ��p�@^{BsG�Ce�{                                    By�  �          @�(��}p��Q�@VffB\  Cu��}p���p�@q�B���Cg�                                    By &  �          @�  �u�!G�@c�
BSp�Cy�u��ff@�=qB���Cn@                                     By�  �          @�  �����H@x��B^Cu��������@��
B��Cg�                                    Byr  �          @�  �����@�z�B�  Cd�{���W
=@��\B���C>:�                                    By,  �          @��׿�33��=q@�p�B���CeO\��33�L��@��B��C>
                                    By:�  �          @�Q쿱����@�  Be��Cm
��녿���@���B�=qCZT{                                    ById  �          @�{�����   @hQ�BJ�
Cmc׿��Ϳ��@�(�Bw  C_�)                                    ByX
  �          @��R���H�>�R@S�
B3�RCs�{���H��@{�Bdz�Cj��                                    Byf�  T          @��Ϳ}p��AG�@X��B;p�C{ٚ�}p��
=@���Bo�
Ct��                                    ByuV  3          @��\��{�@��@QG�B6(�Cy�{��{�Q�@z=qBi��Cr��                                    By��            @w����Ϳ���@[�B�.Ca{���;u@fffB�8RC@c�                                    By��  �          @��\��\)�:�H@B�\B){Cp����\)�ff@i��BW��ChO\                                    By�H  �          @�p���{�HQ�@*�HBp�Cv�쿮{���@VffBJG�Cp}q                                    By��  �          @�G��}p����
?�33A��
C��}p���33@-p�B�HC�.                                    By��  �          @�Q�0������?�33Aw33C��0������@#�
A��HC��3                                    By�:  �          @��\�(�����H?�
=Axz�C�H��(����33@%A�33C��                                    By��  �          @��׿L�����R?У�A�C����L����p�@4z�A�{C�+�                                    By�  �          @�녿&ff��z�@   A���C�` �&ff����@J=qB
�\C��                                    By�,  �          @�  �z�H���H@�HA���C�LͿz�H��z�@_\)B��C�Z�                                    By�  �          @��ͿTz����@,(�A��C�⏿Tz��vff@l(�B,��C��                                    Byx  �          @��
�Q���ff@8Q�B�HC���Q��j=q@uB6�C��=                                    By%  �          @��R�5��
=@,(�A�p�C�}q�5�n{@j=qB/�HC���                                    By3�  T          @���8Q����?��
A�(�C���8Q����\@#33A��C��R                                    ByBj  �          @�����ff��33?�z�A��C�
=��ff�q�@7�B�
C���                                    ByQ  �          @�{����L��@Q�B�C��
����$z�@C33BD�
C�s3                                    By_�  T          @l�Ϳ���p�@HQ�Bo��C|�q���xQ�@\(�B�u�CqE                                    Byn\  �          @�녿@  �   @HQ�BHz�C}T{�@  ��(�@g�BzffCvp�                                    By}  �          @�ff����:�H@J=qB5\)Cy}q�����@o\)Be��Cr                                    By��  �          @�����@��@5B%��Cy8R�����@\��BV�Csk�                                    By�N  �          @�33�@  �u�@�A�  C�|)�@  �L(�@K�B0\)C�k�                                    By��  �          @�녿���xQ�@"�\B{C~�ῇ��L��@VffB333C{��                                    By��  �          @��\����[�@3�
B  C|k�����-p�@`��BI�Cx!H                                    By�@  �          @��ÿ��\�J�H@=p�B#�RCx)���\��H@fffBR��CrE                                    By��  �          @�p����G
=@a�B7��Cuk����\)@�z�BeG�Cm�
                                    By�  �          @��ÿ�Q����R@��A�G�C~+���Q��dz�@QG�B#��C{��                                    By�2  �          @����  ��Q�?�p�AO\)C�� ��  ���@�Ȁ\C�/\                                    By �  �          @�Q쿋�����?�{A:�HC�&f�������@��A��C���                                    By~  �          @�
=���H��
=?�z�A��C�*=���H��p�@=p�BffC~��                                    By$  �          @��ÿ�\)��@A���C~z`\)��33@G�B	��C|��                                    By,�  �          @�zῴz����?��A��HC~�쿴z���=q@=p�A�(�C|ٚ                                    By;p  �          @�p��\��ff?�z�A�{C}�q�\���R@0  A��C|(�                                    ByJ  T          @�p���
=��{?�(�Apz�C|{��
=��  @#�
Aי�Cz�                                    ByX�  �          @�33��������?\)@�Q�C�q��������?�(�A�G�CT{                                    Bygb  �          @�ff�����z�?��\AN�HC�B�������@=qA�ffCz�                                    Byv  �          @�녿�
=��ff?��A�  C|!H��
=��p�@=p�A�=qCzB�                                    By��  �          @��R��{���\@{A���C��{���@P��B�C}&f                                    By�T  �          @�p��!G����\?���A��C����!G����@>{A�C�t{                                    By��  �          @�(����\����@��A��
C������\��z�@]p�B�C���                                    By��  �          @��ͿaG����\@
=A���C�Q�aG����R@\(�B��C��
                                    By�F  �          @���5����@   Aȏ\C�5ÿ5����@dz�B��C��\                                    By��  �          @�=q�}p���{@��A\C����}p���=q@[�B  C���                                    Byܒ  �          @��
��(���Q�@�
A��C�z῜(�����@W
=B��CY�                                    By�8  �          @�(���  ���
@$z�A�(�C�*=��  ��
=@eB��C~n                                    By��  �          @�ff����Q�@Q�A��\C~�H������@Z�HB�C}                                      By�  T          @��ÿ����=q@��A���C}�{�����ff@\(�B�C{�)                                    By*  T          @�G��������@p�A�{CzJ=������R@P��B�RCx�                                    By%�  �          @�����=q���@   A�=qC{W
��=q��(�@Dz�A��Cy}q                                    By4v  T          @�33�ٙ���33?�p�A�p�C}�ٙ���(�@5A��
C{�)                                    ByC  �          @�����G���33?�(�Aap�C|�{��G���@%A�z�C{E                                    ByQ�  �          @�{���
��G�?�\)A�z�C��
���
����@=p�A���C�5�                                    By`h  
�          @���z�H���
?��A��C�Ϳz�H��z�@9��A�C���                                    Byo  �          @�ff�xQ����H?��
A�C���xQ����@8Q�A癚C���                                    By}�  �          @Å�s33���?���A�p�C�G��s33��  @>{A�
=C��R                                    By�Z  �          @����  ���
?�ffAr�\C��R��  ��{@(��A�Q�C���                                    By�   �          @�(���\)���?���AYp�C�]q��\)��
=@�RA�
=C��{                                    By��  �          @��\��G���=q?���A[�
C�⏿�G���@p�A�=qC���                                    By�L  �          @��R�������R?���AL��C����������\@�A�Q�C�S3                                    By��  �          @�  ���\��  ?���AO�C��{���\���
@p�A�\)C���                                    By՘  �          @��R�����?�A]�C��\�����G�@ ��AǙ�C�+�                                    By�>  �          @�\)��G����
?p��AG�C��)��G���=q@z�A��RC���                                    By��  �          @�G���
=���R?Tz�@�C�����
=��{?�p�A�33C��                                    By�  �          @�\)�p������?�{A*�\C�g��p����
=@p�A��C�"�                                    By0  �          @���u���?�{A*�HC�O\�u���@p�A�p�C�
=                                    By�  �          @��
��  ��p�?�{A/33C�H��  ��33@(�A�
=C���                                    By-|  �          @�Q�u���H?��
Ap�C�O\�u��G�@Q�A�  C�\                                    By<"  �          @�G��s33��(�?z�HAG�C�g��s33���H@A��C�*=                                    ByJ�  �          @�
=�z�H���\?O\)@�=qC�=q�z�H��=q?�z�A��C�                                    ByYn  �          @���p�����
?8Q�@�z�C�o\�p����(�?���A�p�C�=q                                    Byh  "          @�녿�33��  ?�A�{Cz�R��33���@3�
A�{Cys3                                    Byv�  �          @�
=����p�?�G�A�p�C|{�����@2�\AծCz�3                                    By�`  �          @�  �ٙ���Q�?�(�A�
C}�\�ٙ����H@1G�A�z�C|T{                                    By�  �          @�(���Q����?���AG33C}�)��Q���z�@�A��C|��                                    By��  �          @�녿������?��AH��C��
�����p�@
=A�C�T{                                    By�R  �          @��׿c�
��33?�ffA!p�C����c�
���@�A���C�n                                    By��  T          @�  �^�R��(�?^�RA��C���^�R���
?�
=A�ffC��q                                    ByΞ  T          @�Q�W
=��>�(�@��C��3�W
=��  ?�G�Ah��C��3                                    By�D  �          @��׿@  ���R>�
=@���C�n�@  ����?�  Ag\)C�Q�                                    By��  �          @�Q�E���{?�\@��HC�T{�E���  ?�=qAs�C�5�                                    By��  �          @���@  ��p�>�@�33C�c׿@  ���?��Ao33C�Ff                                    By	6  �          @��H�333��G�>�p�@i��C��3�333��(�?�z�A_\)C�y�                                    By�  �          @���!G���ff>�=q@*=qC��{�!G����?��AN�HC��                                     By&�  �          @�p��!G���(�>L��@33C��3�!G���  ?��HADz�C��                                     By5(  T          @�\)�5��>�=q@*=qC�s3�5��G�?��\AMG�C�]q                                    ByC�  �          @���8Q���>�Q�@hQ�C�]q�8Q�����?�{A\  C�C�                                    ByRt  �          @��Ϳ0�����H>�z�@>�RC�� �0�����R?��
AP��C�h�                                    Bya  �          @���.{���>aG�@�C���.{���?���AC�C�y�                                    Byo�  T          @��R��R��p�>.{?�Q�C����R���?�z�A;33C��{                                    By~f  �          @�  �+���ff>.{?�33C����+����H?�z�A9��C���                                    By�            @�\)�!G���{<�>��C��{�!G���33?�G�A"�RC�Ǯ                                    By��  T          @�\)��z���
=���
�B�\C�����z���(�?xQ�A�C��                                    By�X  
�          @��������ý�\)�&ffC�j=����ff?n{A33C�aH                                    By��  T          @�Q���H��������C�����H���?uA��C���                                    ByǤ  T          @�G�����Q�<��
>#�
C�q����?�  A�RC�g�                                    By�J  �          @�G��
=q��Q�=�G�?��C�P��
=q���?�=qA,Q�C�C�                                    By��  �          @�ff���H���Ϳ����RC�����H���>�G�@�
=C��f                                    By�  �          @�G��:�H���Ϳ����#�C�}q�:�H�����\)�#�
C���                                    By<  �          @��R�#�
���׿�ff�JffC��3�#�
������
�EC��f                                    By�  �          @�z�O\)���R��Q��:�HC��3�O\)���\�k����C��                                    By�  T          @�{�s33��녿=p���C�(��s33���>L��?�(�C�33                                    By..  �          @�ff�����=q����p�C��)�����=q?�\@��C���                                    By<�  T          @��H��\)��\)����\)C��쿏\)��p�?\(�A�\C��                                    ByKz  T          @ƸR��z����H>���@@��C�� ��z���ff?��AC�
C��H                                    ByZ   �          @�=q���ƸR>B�\?�
=C��=�����H?�Q�A-�C���                                    Byh�  �          @θR�����˅��\)�+�C�ῌ����G�?uA(�C��                                    Bywl  �          @θR����˅>.{?�  C��f����Ǯ?�Q�A)C��\                                    By�  �          @�\)��\)��(�>#�
?���C����\)�ȣ�?�
=A'\)C���                                    By��  �          @�Q��  ��=q>8Q�?���C�N��  �ƸR?�
=A'�
C�1�                                    By�^  �          @�
=��\)��{>�\)@ ��C}5ÿ�\)���?�G�A4  C|��                                    By�  �          @�
=������G�>�\)@��C���������p�?��\A4��C�h�                                    By��  �          @�{�����ʏ\>W
=?�33C��׿����ƸR?��HA,��C���                                    By�P  �          @�ff������=�\)?
=C�%�����\)?�ffAffC��                                    By��  �          @�
=��
=�˅=L��>�
=C�����
=�ȣ�?��A�C���                                    By�  �          @�zῇ��ə�<�>�  C�<)�����
=?�  Az�C�,�                                    By�B  �          @��Ϳ�{�ə�    <��
C�  ��{��\)?xQ�A  C��                                    By	�  t          @�������H>�  @��C�\)�����
=?�(�A.�\C�G�                                    By�  �          @�����
��=q>L��?�\C�aH���
�ƸR?�z�A'
=C�N                                    By'4  �          @�  �G���ff�L�Ϳ�C�o\�G����?=p�@ڏ\C�h�                                    By5�  �          @�  �(����ff�u�z�C�Ф�(����p�?(��@�=qC���                                    ByD�  �          @��ÿ�R����W
=��\C���R��ff?0��@�=qC�H                                    ByS&  �          @�G��\)�����ff��=qC�S3�\)���>�ff@�  C�S3                                    Bya�  �          @�������!G�����C�q�����R>��@�RC�u�                                    Bypr  �          @����(����5�أ�C����(���
=>8Q�?�p�C���                                    By  T          @���!G������c�
�\)C��f�!G����
����\)C��\                                    By��  �          @���   ���\�G���\C��Ϳ   ��(�=�Q�?Tz�C���                                    By�d  �          @����G���z�u�Q�C�ٚ��G���
=��Q�fffC��                                     By�
  
�          @�(��������ÿ��
�ffC�n�����Å�\)����C�s3                                    By��  �          @�G��L�������  �7�
C���L���ȣ׾�33�L(�C�{                                    By�V  �          @�Q��
=�Å����@��C�;�
=��\)��
=�tz�C��                                    By��  �          @�Q������H�����B=qC�q�����
=��(��|��C�~�                                    By�  �          @�  ��  �\���Q�C��{��  ��
=�
=q��{C���                                    By�H  �          @�\)���
���ÿ��
�c�C�uþ��
��{�(���ÅC��                                     By�  T          @ʏ\��G���z����a��C�|)��G���G��(����  C�~�                                    By�  �          @�=q>k����
�Ǯ�dQ�C�{>k��ȣ׿.{��
=C�                                    By :  �          @��=L����(����H�V�HC�AH=L�����ÿ����p�C�@                                     By.�  �          @�ff���
��\)����u�C�논��
���ͿL������C���                                    By=�  �          @�G�>���\�Ǯ�e�C��
>���Ǯ�333��{C��3                                    ByL,  �          @��=#�
��ff������
C�*==#�
��G��L�Ϳ�{C�(�                                    ByZ�  �          @�33=����ȣ׿��
�p�C�u�=�����33�.{��  C�s3                                    Byix  �          @�p�>�\)���R�����k�C�XR>�\)���
�=p���C�O\                                    Byx  �          @�
=>�z���  �����n�HC�b�>�z�����G���C�Y�                                    By��  �          @���>�=q���Ϳ�(��333C�Ff>�=q��Q�\�_\)C�AH                                    By�j  �          @�{>�z����ÿ���H��C�e>�z��������(�C�]q                                    By�  �          @�\)>8Q���\)����{C��f>8Q�����fff�
{C��                                     By��  �          @�G�>#�
��녿�\)�x��C��f>#�
��\)�W
=���RC��H                                    By�\  �          @�z�>8Q���������F�RC��
>8Q��Å�����
C��3                                    By�  �          @\=�Q���G��.{���C�h�=�Q��\=�?���C�h�                                    Byި  �          @���>8Q���zῨ���IC���>8Q���Q�
=q��z�C��
                                    By�N  �          @��
>��R��
=��=q�I�C�� >��R���H�������C�xR                                    By��  �          @Å>�{����}p���\C��)>�{��=q�aG���C��
                                    By
�  �          @�{>�
=�����\���\C�q>�
=���H���
�   C��                                    By@  T          @��H>�����������C�q�>����{���H�>�RC�\)                                    By'�  �          @�p�?
=q������=qC�˅?
=q���R�����X��C��\                                    By6�  �          @�=q>�=q�ƸR�xQ��=qC�>�>�=q���þ8Q��C�:�                                    ByE2  �          @���?J=q�����'���p�C��f?J=q��=q�����ffC��R                                    ByS�  �          @�G�?Tz�����2�\��C��?Tz������\���C���                                    Byb~  �          @��>�33��{����ffC��=>�33���Ϳ�=q�C33C��)                                    Byq$  �          @�  �#�
��p����\��HC�Ф�#�
�Ǯ�����C�Ф                                    By�  �          @�녿(���  ?   @�(�C��ÿ(���z�?�AA�C�Ǯ                                    By�p  �          @�������33?z�@��C��Ϳ�����?�p�AQ��C���                                    By�  �          @�  ��R���R>���@HQ�C�Ǯ��R��(�?uA ��C��)                                    By��  �          @�������Q�>.{?�G�C������ff?Tz�A
{C��                                    By�b  �          @��R�!G�����>�z�@8��C���!G���=q?uA�C�Ǯ                                    By�  �          @�
=�.{���?(��@ӅC����.{��  ?���AV=qC�~�                                    By׮  T          @��H��33��\)�u�=qC�*=��33���������G
=C�0�                                    By�T  �          @�33<��
��ff�G����
C�
<��
���Ϳ�33�g�C��                                    By��  �          @��ͼ#�
��z�����\)C��#�
��(���(����C��                                    By�  �          @�\)�#�
����   ��C�*=�#�
������{�\��C�1�                                    ByF  �          @�(�����  �z���  C��
������=�Q�?^�RC���                                    By �  �          @����ff���R?��AZ�\C��H��ff����@�\A��C���                                    By/�  �          @�z���H���?�A�{C�� ���H���@p�A�Q�C�b�                                    By>8  �          @�z�
=q���\>��
@J�HC�` �
=q��Q�?z�HA�\C�W
                                    ByL�  �          @�p��(���(�=�Q�?fffC���(����\?B�\@��HC�H                                    By[�  �          @��Ϳ   �����
=���C��ÿ   ���
>u@�
C��
                                    Byj*  �          @��ͽ�Q�����Ǯ�uC����Q����
>��@#�
C��                                    Byx�  �          @�z�����H=L��?�\C�h�������?333@�33C�e                                    By�v  �          @��R��G����\?=p�@��C�)��G����R?���AVffC��                                    By�  
�          @��׿�����\?}p�A  C��\�����{?У�Az�HC���                                    By��  �          @�
=��  ��G�?���A%C�R��  ��z�?ٙ�A�(�C��R                                    By�h  �          @��Ϳ�ff��\)?n{A��C��쿆ff��33?��Ar�\C��3                                    By�  �          @��
��{��{?c�
A�C�z῎{��=q?�  Alz�C�Z�                                    Byд  �          @����ff��Q�?G�@�=qC��R��ff��z�?��AYC��q                                    By�Z  �          @��ͿxQ����?s33A��C�8R�xQ���33?ǮAu�C��                                    By�   �          @���fff��  ?xQ�A�C����fff���
?���Aw�C�u�                                    By��  �          @��\�O\)���?!G�@�\)C�  �O\)��z�?�p�AC33C��                                    ByL  �          @��׿�  ��(�?L��@�C��쿀  ��Q�?���A]G�C��R                                    By�  �          @��H��G���p�?z�HA�C�𤿁G���G�?ǮAw�C��3                                    By(�  �          @�G��E���?J=q@�C�H��E���=q?�33AV�RC�7
                                    By7>  �          @�z�^�R��Q�?^�RA�\C��)�^�R��z�?�p�A_�C��f                                    ByE�  �          @�(��Tz���\)?�G�A��C��Tz����H?�\)AuG�C���                                    ByT�  �          @�ff�h����Q�?��HA4��C��\�h����33?�A�Q�C��3                                    Byc0  �          @\�
=���\?���At  C���
=��z�@�A�  C�                                      Byq�  �          @�G��G���?�z�A�33C��G����R@p�A���C��3                                    By�|  �          @�G��(����?�
=A�33C����(�����R@\)A\C��                                    By�"  �          @���W
=����@ffA�  C����W
=��=q@(��A���C���                                    By��  �          @�ff�^�R���?��A��HC��\�^�R���H@�A�\)C�b�                                    By�n  �          @�Q���\����?��A'�Cz�)��\���?ٙ�A{�
Cz+�                                    By�  �          @�\)�p���ff?�
=A.�\Cx��p����?�p�A��RCxG�                                    Byɺ  �          @�Q������R?�
=AS�
Cy�f����G�?�p�A�G�Cy
                                    By�`  �          @��������ff?�(�AX(�Cx���������@ ��A���CxO\                                    By�  �          @�G��\)��z�?�33Ar�HCx\)�\)��ff@
�HA��Cw��                                    By��  �          @ə���
���?�=qA�33Cws3��
���
@ffA�33Cv�3                                    ByR  �          @���'�����@�A�Q�Cs�{�'�����@*=qAȸRCr�\                                    By�  �          @��
�5��z�@p�A�=qCq)�5��z�@;�AۅCo޸                                    By!�  �          @�=q�,�����@G�A�z�Cr���,����Q�@0  A�(�Cq��                                    By0D  �          @ə���R���
@�A�(�Cu0���R����@*�HA�ffCt@                                     By>�  �          @�녿�G���  ?�z�A���C|�q��G�����@�A���C|h�                                    ByM�  �          @Ǯ������H@�A�(�C{J=�����(�@!�A���Cz��                                    By\6  �          @Ǯ���R��G�@33A��CzLͿ��R���H@#33A��RCy�
                                    Byj�  f          @�  ��(���=q@�\A�Q�Cz����(����@!�A�33Cy�
                                    Byy�  �          @�  �G�����@�\A�(�Cz��G���33@!G�A��\CyW
                                    By�(  �          @�
=�
=��Q�?�(�A��Cy��
=���@��A��
CxO\                                    By��  �          @�\)�\)��p�@�A�z�Cw�{�\)��
=@#33A�Cv�=                                    By�t  �          @�=q�.�R��  @\)A��
Cp{�.�R��Q�@9��A�z�Cn�=                                    By�  �          @�=q�6ff��=q@
=A���Coc��6ff���H@1G�A��Cn.                                    By��  �          @��H�B�\���@{A��Cl޸�B�\��@7
=A�z�Ck��                                    By�f  �          @����C33���R@�RA��\Cm��C33��\)@7�A�\)Ck�                                     By�  �          @�33�G
=���@   A�Ck�q�G
=��(�@8Q�A�  Cj�)                                    By�  �          @��H�3�
���@�HA�{Co�R�3�
���H@4z�A�p�Cn��                                    By�X  �          @����#33���@	��A��
Cs���#33���R@%�A�Q�Cr                                    By�  �          @����(Q���(�@{A�=qCq���(Q����@7�A��
Cp��                                    By�  �          @��
�U��mp�@`��B�Cd��U��Z=q@s�
B��Ca�f                                    By)J  �          @���Fff��ff@5�A�Ci���Fff�|��@J�HB �
Cg޸                                    By7�  �          @�  �I���p��@:=qA�\)Cf��I���`��@Mp�B�HCd�                                    ByF�  �          @���*=q��Q�@\)A�G�Cn��*=q��G�@5�A���Cl�f                                    ByU<  �          @��������(�?�{A�G�Cr�R������R@�RA�(�Cq�)                                    Byc�  �          @�p��!G����
@	��A��CqaH�!G���@ ��A�Cp^�                                    Byr�  �          @��\�z�����@(��A�z�Cp��z��s�
@<��B  Cn��                                    By�.  �          @�����R���@Q�A�Cq�)��R�}p�@,��A�Cp��                                    By��  �          @�
=�>{����?�p�A�p�Cj^��>{�~�R@33Aƣ�Ci:�                                    By�z  �          @����.�R�p��@p�A�33Cj  �.�R�c33@0  A�  ChxR                                    By�   �          @�  �!��aG�@�RA�ffCjE�!��Tz�@0  B�\Ch�H                                    By��  �          @��'��W
=@{A��Ch��'��J=q@.{B�Cfff                                    By�l  �          @���,(��\(�@=qA��Cg���,(��P  @*�HBG�CfW
                                    By�  �          @���*=q�J�H@�A�Cf
=�*=q�>�R@*�HB��Cd=q                                    By�  �          @����0���#�
@/\)B(�C^���0���@:�HB�C\O\                                    By�^  �          @����;��P��@,(�A�  Cc�q�;��C33@:�HB
��Cb�                                    By  �          @�p��:=q�Z�H@#�
A�Ce�{�:=q�N{@3�
B��Cc�)                                    By�  �          @�z��?\)�r�\@�\A��Cg���?\)�g
=@$z�A噚Cf\)                                    By"P  �          @����@  �R�\@@��B33Cc�)�@  �C�
@P  Bz�Ca��                                    By0�  �          @�(��Fff�]p�@+�A�=qCd)�Fff�P  @;�Bp�Cb\)                                    By?�  �          @�(��Dz��fff?�Q�A���Ce���Dz��\��@��A�CdL�                                    ByNB  �          @�{�:=q�g
=@z�A�ffCg#��:=q�\(�@$z�A�  Ce��                                    By\�  �          @����.{���
?�\)A�Q�Cl�
�.{�~�R@
=qA���Ck��                                    Byk�  �          @���(Q����?��RA���Cm
�(Q��z=q@G�A�  Cl
                                    Byz4  �          @�z��/\)�s�
@�\A�(�CjB��/\)�j=q@�
A�Q�Ci!H                                    By��  �          @��J�H�HQ�@'�A��
C`�3�J�H�<(�@5B{C^��                                    By��  �          @��H�Mp��C33@\)A���C_��Mp��7�@,��B =qC]�                                    By�&  �          @��\�:�H�Fff@��A��Cb��:�H�<(�@{A�  Ca!H                                    By��  �          @�z����W�@�\A�\)Cj=q���N{@G�A���Ci�                                    By�r  �          @�
=�G��Tz�?��A��HCn���G��L��?�{Aљ�Cm�R                                    By�  �          @�\)�33�QG�?޸RA�=qCm�R�33�I��?��HA�z�Cl�3                                    By�  T          @�G��z��R�\?�A�  Cm�\�z��J=q@�A�  Cl                                    By�d  �          @��	���[�?�\A��Cm�3�	���S�
@   A�
=Cl�R                                    By�
  �          @�
=�&ff�^{?�
=A�{Ci&f�&ff�Vff?�z�A�ffCh.                                    By�  �          @��R�+��=p�?�p�A��Cc� �+��4z�@
�HA�G�Cbc�                                    ByV  �          @����1��#�
?�p�AܸRC^���1���@��A���C])                                    By)�  �          @g
=�\)���?�=qA�p�CY(��\)��G�?���B�\CWL�                                    By8�  �          @Z�H�
=q�
�H?���A���Ca
=�
=q�z�?˅Aߙ�C_��                                    ByGH  �          @tz��{�,(�?�Q�A��Cfz��{�%?�\)A�z�Cek�                                    ByU�  �          @w��(��7�?��
A�(�Ch�f�(��1�?��HA�\)Cg�=                                    Byd�            @�������p��?�G�A��\Cz�H�����j�H?�  A�  Cz33                                    Bys:  �          @��\��z�����?xQ�AD��C~�׿�z���\)?��RA|z�C~��                                    By��  T          @��׿���  ?�{AfffCw^����z�H?�{A���Cv�R                                    By��  T          @��z�H���?�@ƸRC�:�z�H��Q�?Tz�A33C�,�                                    By�,  T          @�{��33��Q�?+�@�  C�0���33���R?uA3
=C��                                    By��  
(          @�녿xQ���p�?(�@�ffC�n�xQ����
?h��A&ffC�aH                                    By�x  "          @�=q�0�����>��R@a�C�
�0�����R?(�@߮C��                                    By�  T          @���\)��
=��  �.{C�:�\)��\)=�\)?=p�C�:�                                    By��  "          @��������þL�����C�J=����G�=�?�  C�J=                                    By�j  �          @���>��H����(�����
C���>��H���׾�{�mp�C��                                    By�  �          @�\)>�ff����0�����C�~�>�ff��{�\��C�z�                                    By�  T          @�
=>����Ϳ(����  C���>�����33�x��C��                                    By\  �          @�Q�!G���
=>��@?\)C�h��!G���{?��@�=qC�e                                    By#  �          @���Tz�����?!G�@���C��=�Tz����?c�
A0��C��q                                    By1�  "          @�p��Y�����?   @��C��3�Y������?B�\A�\C���                                    By@N  T          @�  ��{��=q?0��A�C�0���{����?s33A8(�C�                                      ByN�  
�          @�p���Q���?^�RA,��C  ��Q����
?�\)A_
=C~�\                                    By]�  "          @�ff��p����?���AY��C{���p����?���A�\)Cz�R                                    Byl@  "          @���ff��Q�?�
=Aip�Cy�q��ff��?�A��HCy��                                    Byz�  
�          @��
������?�z�Aj�RCxG������G�?���A��Cw�                                    By��  �          @�\)���H�hQ�?�p�A��Cq����H�a�?�A��
Cq                                      By�2  �          @��R�����k�?�{A���Cr)�����e�?�A�33Cq}q                                    By��  
(          @�녿���z=q?�z�A��Ct0�����u�?�\)A�{Cs��                                    By�~  T          @�������
=?�p�AtQ�CyE��������?��HA�p�Cx�                                    By�$  "          @�
=��=q��G�?�33AaCy�{��=q��
=?���A�{Cy�=                                    By��  
Z          @�
=��p����H?���AW33C{!H��p�����?�=qA��HCz޸                                    By�p  "          @��R�޸R��ff?�(�Aqp�CwxR�޸R���
?���A�33Cw!H                                    By�  T          @���������?�=qA~{Cw�������\)?ǮA�\)Cv��                                    By��  T          @�녿�G���?E�A�
Cyff��G���(�?�G�A8��Cy5�                                    Byb  �          @�����33��\)?#�
@陚Cz�=��33��{?c�
A!�Cz��                                    By  �          @�Q��  ����?\)@�p�Cy^���  ���
?L��A33Cy:�                                    By*�  �          @�=q��
=����?W
=A��C}O\��
=��\)?��AF�HC}#�                                    By9T  �          @��H���
���
?z�HA1C�����
��=q?�p�A_�C��
                                    ByG�  
�          @��\��  ����?fffA#�
C�B���  ��33?�33AQp�C�33                                    ByV�  
Z          @��Ϳ����{?h��A#�
C��������z�?�z�AQ�C��R                                    ByeF  �          @��
�p����\)?5A   C��H�p����{?uA-�C���                                    Bys�  T          @����+���?E�Az�C�  �+���z�?�G�A9��C�
                                    By��  T          @�G���\)��ff?\(�A�C�b���\)����?�{AK33C�]q                                    By�8  �          @��������Q�?   @�p�C�  �����\)?=p�A�
C��                                    By��  T          @���������Q�?�\@���C�  ������\)?@  A��C�q                                    By��  
�          @��
�G���Q�>�
=@��RC��
�G���\)?(��@�RC���                                    By�*  �          @�����R��z�?+�@��C}쿾�R��33?fffA ��C|�                                    By��  �          @��\��Q����\?
=@��
C}n��Q�����?Q�A  C}Q�                                    By�v  
�          @�zῙ����\)>�p�@�p�C�8R�������R?(�@�=qC�/\                                    By�  T          @�����
=>���@i��C~
=����ff?\)@�Q�C}��                                    By��  �          @��׿�{���
?�\@�=qC�����{���H?@  A33C��                                     By h  �          @�녿p�����>#�
?�G�C��q�p������>���@���C�ٚ                                    By   "          @�
=�xQ����
=��
?\(�C�⏿xQ����>���@^�RC��                                     By #�  
�          @��R�z�������R�N{C���z���p��u�
=C��3                                    By 2Z  "          @�{������(������\C��׾������;�\)�=p�C��                                    By A   �          @���>����p���z��A�C�Ǯ>����
=�h���\)C��f                                    By O�  T          @�G�>#�
�����Q��EC��{>#�
���R�p���(�C��3                                    By ^L  
�          @���>B�\�����{�8��C���>B�\���R�\(���C��)                                    By l�  T          @�Q�#�
��{�Q��	��C�Ф�#�
��
=�z�����C�Ф                                    By {�  �          @�{�aG���z��R��=qC��3�aG���p��\����C��{                                    By �>  �          @���\)���Ϳ
=q��p�C�}q��\)��p������H��C�~�                                    By ��  �          @��
����녾�  �&ffC�B�����=q�#�
��Q�C�B�                                    By ��  
Z          @��ͿL�����\��p��n�RC��=�L�����H����  C���                                    By �0  �          @�ff�   ��p��L�;�C�z�   ��p�>L��@   C�z�                                    By ��  
�          @��R�#�
��(�>��R@FffC��׿#�
���
?��@��C��H                                    By �|  "          @�������?��RAJ=qC�Uþ����R?�(�AqC�S3                                    By �"  
(          @��?^�R���@ ��A���C��R?^�R����@�RA���C��                                    By ��  
�          @��\>����z�?��A?\)C��\>�����H?�\)Af=qC���                                    By �n  T          @��׿���ff�����\)C�@ ����
=���
�Tz�C�B�                                    By!  
�          @��ÿ(���
=��p��tz�C���(�����\)��C��{                                    By!�  T          @�  ��\���R��\)�=p�C�XR��\���R>#�
?��C�XR                                    By!+`  
�          @�\)�����ff�\)����C��R�����ff=�Q�?n{C��R                                    By!:  �          @�Q��ff��\)=L��?�C����ff��\)>�\)@8��C���                                    By!H�  �          @�G��
=q��  =�Q�?h��C�1�
=q���>��R@P  C�0�                                    By!WR  
�          @�  �B�\��>�  @%C���B�\��p�>��@��C�                                    By!e�  
�          @�  �8Q���ff>8Q�?�33C�5ÿ8Q���{>���@��RC�4{                                    By!t�  
�          @���333��{=�Q�?h��C�Z�333��>��R@Mp�C�Y�                                    By!�D  	�          @�
=�����{>\)?��RC�R�����>�Q�@r�\C�
                                    By!��  "          @�ff��p���p������RC�\��p���p�>B�\?�p�C�\                                    By!��  
�          @�{�8Q���>B�\@G�C���8Q���p�>��@��C��                                    By!�6  
�          @��R������>���@�{C�AH�������?(�@�ffC�@                                     By!��  
Z          @��;Ǯ��(�>B�\@ ��C��f�Ǯ���
>���@�Q�C��                                    By!̂  "          @�=q�\�����\)�ǮC��\�\���=u?0��C��\                                    By!�(  "          @�(��B�\�����Q��uC����B�\���
�������C�                                      By!��  �          @���L����=q�z��ȣ�C��;L�����H�\��G�C���                                    By!�t  
�          @�  >\)��
=����=qC��
>\)��\)��  �&ffC��
                                    By"  
�          @�  >�z���
=�����Q�C���>�z�����L���33C���                                    By"�  
�          @���>�Q���\)�z���=qC��f>�Q���  ��p��w�C��                                    By"$f  B          @�  >�  ��zῆff�0��C�U�>�  ���Y���C�S3                                    By"3  "          @���k���G��(����HC��)�k���녾����p�C��q                                    By"A�  	`          @��
�aG�����>���@J�HC�G��aG���Q�?   @��C�C�                                    By"PX  �          @����G���p�?n{AC�!H��G���(�?�\)A?\)C�{                                    By"^�  �          @�Q�k����>��@��
C�1�k���z�?+�@߮C�.                                    By"m�  �          @��ÿ8Q����R?   @�C�<)�8Q���{?333@陚C�7
                                    By"|J  �          @�G��c�
���?8Q�@�C�Uÿc�
��(�?k�Ap�C�N                                    By"��  T          @���Y�����
?B�\@�\)C��f�Y�����H?uA!G�C��                                     By"��  �          @�\)�
=��{>�33@j�HC����
=��p�?��@���C��f                                    By"�<  �          @�  ��\��ff>�ff@�  C�P���\��?(��@�33C�N                                    By"��  �          @�ff>B�\��������C�>B�\������-p�C�                                    By"ň  �          @��R����{��z��@  C��{����ff��Q�h��C��{                                    By"�.  �          @�  ������ff�(���33C�~�������
=�������C��                                     By"��  �          @����������þǮ����C�uý��������B�\��(�C�u�                                    By"�z  �          @�Q�\���R=L��>��HC��\��ff>u@#�
C�                                    By#    �          @�(�>aG��������;\)C�:�>aG����ÿfff�ffC�8R                                    By#�  �          @��R?W
=�����\)C��?W
=��Q����33C��                                    By#l  T          @��
?}p������(Q���C���?}p������R��C�b�                                    By#,  �          @�33?��������   ��G�C�(�?�����(��ff��p�C�f                                    By#:�  �          @��H?c�
��\)�{��33C���?c�
������
��G�C���                                    By#I^  �          @�p�?aG���ff�{��=qC���?aG���G���
��Q�C�z�                                    By#X  T          @�p�?��
��33�&ff��G�C���?��
��{�(���C���                                    By#f�  �          @�{?xQ����H�)����{C�E?xQ����\)��Q�C�%                                    By#uP  �          @�?�G�����"�\��{C�o\?�G���  �Q���ffC�P�                                    By#��  �          @�?O\)��Q�����\C���?O\)��=q�����RC��f                                    By#��  �          @��?5��\)�����t��C��?5���ÿ����U�C��                                    By#�B  �          @�>aG���G���=q�@��C�:�>aG����\�fff� ��C�8R                                    By#��  �          @�ff=��
��p���(����C�s3=��
��{��  �0  C�q�                                    By#��  �          @�����
��
=��z��J=qC��콣�
��\)��G���
=C���                                    By#�4  �          @��=u��
=�����O\)C�XR=u��
=��G���G�C�XR                                    By#��  �          @�(���G���33=��
?fffC�����G����H>��@7�C��f                                    By#�  �          @�=q��33����>�ff@���C�׾�33��Q�?�R@��C�H                                    By#�&  �          @�Q�
=q��{>�p�@���C���
=q��p�?
=q@�Q�C��R                                    By$�  T          @��R�0�����?xQ�A4(�C�녿0������?���AS33C���                                    By$r  �          @����ff���\?5A�C�S3��ff���?aG�A%G�C�O\                                    By$%  �          @��þ������>�Q�@�33C�
������\)?�@���C��                                    By$3�  �          @�=q    ����Q��C�H    ��Q�&ff����C�H                                    By$Bd  �          @�(��#�
���H����
=C����#�
�����Q�����C��                                     By$Q
  �          @�=�����=q���\�5��C���=�������Y���ffC��\                                    By$_�  �          @�z�>�����׿�ff�<  C��)>����녿^�R���C���                                    By$nV  �          @�ff    ���8Q��33C���    ��{�#�
����C���                                    By$|�  �          @��>#�
��G������z�C��f>#�
�����u�,��C��                                    By$��  �          @���>L����\)��R���C�!H>L����  ��ff���
C�                                      By$�H  �          @�
=>�
=���H���\�=C�y�>�
=���
�\(���HC�t{                                    By$��  �          @��R>Ǯ���
�Tz��p�C�L�>Ǯ��z�(�����C�H�                                    By$��  T          @�>\���;aG��!G�C�>�>\���ͽL�Ϳ
=C�>�                                    By$�:  �          @�{>�33����u�0  C�f>�33�����\)�Q�C�                                    By$��  �          @�ff>���(��\)�ҏ\C���>���z��������C��                                    By$�  �          @��\?�����=p��ffC�E?���Q����\)C�AH                                    By$�,  �          @��
?k���(������ZffC�XR?k��������<  C�N                                    By% �  �          @�G�?�G������{�z�RC��?�G����ÿ����\z�C���                                    By%x  �          @���.{��녿(�����RC�  �.{���\���H��G�C��                                    By%  �          @�(��n{���׾�(����HC��H�n{���þ�=q�L��C��                                    By%,�  �          @��
�u����   ��(�C�K��u��  ��{�~�RC�O\                                    By%;j  �          @�p����
���׿�R��RC��Ϳ��
���þ����\C��3                                    By%J  �          @�{�������ÿ+���=qC�������������\��{C���                                    By%X�  �          @�G���Q����ÿ&ff��{C}J=��Q��������H���HC}Y�                                    By%g\  	N          @�G������׿5�G�C}c׿���G������\)C}u�                                    By%v  T          @�\)�\(���녿Y�����C���\(����H�0����ffC���                                    By%��  �          @�
=�\)����Tz��=qC��3�\)��z�+���
=C��
                                    By%�N  "          @�\)��G���(��Tz��C�n��G����Ϳ(����C�q�                                    By%��  �          @�  ��\��z�\(��=qC���\����0�����RC��                                    By%��  �          @��׿+����ͿTz��(�C���+���p��(�����HC�#�                                    By%�@  T          @��\�����  �O\)��C��H������׿#�
����C���                                    By%��  "          @��׾�(����Ϳn{�+\)C�y���(����B�\�z�C�}q                                    By%܌  
Z          @��\�h����z�z�H�2�HC��
�h�����Q��Q�C���                                    By%�2  T          @��������J=q��RC�������Q��R�߮C�                                    By%��  �          @��
=�\)�����{�H��C�ff=�\)���ÿp���)C�e                                    By&~  �          @�33�.{��Q�c�
�!C���.{���ÿ8Q��ffC��                                    By&$  T          @���z���=q�����p�C�\)�z���(���(����C�g�                                    By&%�  T          @��׿c�
���
�G���{C�AH�c�
��{���ȸRC�Y�                                    By&4p  
�          @�녿�
=����	�����HCt�{��
=��� ������Cu5�                                    By&C  T          @�녿333����G���33C�|)�333����
=�ǮC��\                                    By&Q�  
�          @������ff� ����33C��=�����׿�{���C��
                                    By&`b  
�          @��ÿ(����Ϳ�\��=qC�Ϳ(����R��\)���\C��                                    By&o  
�          @�{�\���\�޸R��Q�C�~��\��z�˅���\C��f                                    By&}�  T          @���G���녿�\��\)C�R��G����
��\)����C�"�                                    By&�T  
�          @��.{���
��(�����C��ÿ.{����������
C���                                    By&��  T          @�Q쾮{���
��  �B�\C�޸��{����W
=�"�HC��                                    By&��  "          @��׿E���p������C�Q�E���{��Q����HC�W
                                    By&�F  
�          @�\)��R���H�333�	p�C�&f��R����
=q��33C�+�                                    By&��  �          @�
=������\)���H��=qC�Z�������ÿ�ff��(�C�]q                                    By&Ւ  
�          @���>\�����33�^�HC�Z�>\���Ϳ}p��>�RC�U�                                    By&�8  
�          @�33>�
=��{��z��]��C���>�
=��
=�}p��=p�C���                                    By&��  
�          @�Q�>��
�����33�_33C�H>��
���Ϳz�H�?
=C��q                                    By'�  
�          @�=�G����
�333�	G�C��=�G���z���љ�C��                                    By'*  �          @��>�����G��c�
�0��C�޸>�����=q�:�H�Q�C��)                                    By'�  
�          @��
>aG����ÿW
=�0��C�s3>aG���녿.{�(�C�p�                                    By'-v  �          @��H>����Q�+��C��H>�����ÿ���=qC��q                                    By'<  �          @��\?!G���Q��G���=qC�/\?!G����׾�z��r�\C�+�                                    By'J�  �          @�p�?Y�������   ��G�C��{?Y����=q��33����C���                                    By'Yh  
�          @�z�?�  ���׾L���(Q�C���?�  ���ýL�Ϳ�RC��\                                    By'h  �          @�(�?fff���׾����y��C��R?fff���þ\)��\)C���                                    By'v�  
�          @��
?fff��Q쾀  �O\)C��q?fff���׽�Q쿙��C���                                    By'�Z  "          @�G�?�����녿=p���HC��f?������H�
=��p�C���                                    By'�   �          @��H?������ÿW
=�3�C�J=?�����녿333��C�:�                                    By'��  "          @��\?�ff���ÿ^�R�9�C���?�ff��녿8Q���C��f                                    By'�L  �          @���?�33��G��Tz��2{C���?�33��=q�.{���C��                                    By'��  �          @��H?�33���
�=p��{C���?�33��z�
=���\C���                                    By'Θ  �          @�Q�?����{������h��C���?����}p��k��HQ�C�t{                                    By'�>  �          @���?�����H�O\)�-�C�!H?�����
�(�����C��                                    By'��  �          @��?Q����\)��  C��?Q���ff������z�C���                                    By'��  �          @��?z�H���Ϳ
=��(�C���?z�H��p���(���G�C��
                                    By(	0  T          @���?����녿c�
�>�RC��H?�����H�:�H�G�C�t{                                    By(�  �          @��\?�
=��G��   ��33C�?�
=��녾�{���C���                                    By(&|  �          @��?�
=�p  ��33�{\)C��?�
=�r�\��G��[�C��
                                    By(5"  �          @��
?�p��a녿�\)��ffC��f?�p��e��p���
=C�q�                                    By(C�  
�          @��
?�\)�s33��Q����C���?�\)�vff�����\)C�n                                   By(Rn  
�          @�33?�\�l(���(����\C���?�\�o\)��=q���\C���                                   By(a  
�          @��?��R�tz῕�}�C��f?��R�w����\�\��C���                                    By(o�  �          @���>�(��������G�C��>�(���Q쾸Q���=qC�޸                                    By(~`  �          @�  ?���ff��
=���C�` ?����R��  �H��C�]q                                    By(�  �          @��H>�����\����P  C���>�����\���
��  C��f                                    By(��  
Z          @��\>aG���녾�p����HC�g�>aG���=q�B�\�C�ff                                    By(�R  
�          @�G�>�Q���Q쾸Q����C�L�>�Q����׾8Q��{C�K�                                    By(��  T          @�  >��R��\)�k��=p�C��{>��R����L�Ϳ0��C��{                                    By(Ǟ  	�          @��H>������þ�ff��\)C��>���������\)�l(�C���                                    By(�D  	�          @�p�?c�
��  �E��+
=C�B�?c�
���׿(��
=C�8R                                    By(��  "          @�\)?c�
��녿333��
C�&f?c�
���H����\)C�q                                    By(�  
�          @�ff?����w
=��  �\z�C�s3?����y���W
=�9G�C�]q                                    By)6  "          @�  ?�p��q녿��\����C��\?�p��u���\)�s
=C��\                                    By)�  	�          @���?�z��x�ÿ�33�xz�C���?�z��|(��}p��U�C���                                    By)�  
�          @�  ?��
�{���  �Z=qC��?��
�~{�W
=�6=qC��
                                    By).(  �          @�?�Q��z=q�\(��=p�C�� ?�Q��|(��0����C�o\                                    By)<�  
(          @�?���~{�=p��"�RC�~�?����  �����C�q�                                    By)Kt  �          @���?�33���ÿO\)�/33C��)?�33��녿#�
�
=qC��                                    By)Z  
:          @���?}p���33�0���G�C�˅?}p���(�����\)C��H                                    