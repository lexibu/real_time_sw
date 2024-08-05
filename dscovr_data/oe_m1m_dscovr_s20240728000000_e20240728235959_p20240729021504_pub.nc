CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240728000000_e20240728235959_p20240729021504_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-29T02:15:04.328Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-28T00:00:00.000Z   time_coverage_end         2024-07-28T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lByj1   �          @�@Dz�@�
=���R�S�
B^33@Dz�@��0���陚B]33                                    Byj?�  �          @�(�@e�@u��ff�5�B<
=@e�@o\)��\)�mp�B8�                                    ByjNL  "          @��\@hQ�@u�!G����B:ff@hQ�@qG��s33�%G�B8ff                                    Byj\�  �          @��
@u�@mp���ff��G�B0ff@u�@j=q�B�\��RB.��                                    Byjk�  �          @�@n{@z=q��33�k�B9p�@n{@w
=�.{��ffB8(�                                    Byjz>  �          @�@tz�@j=q����9�B/{@tz�@c33��33�o�B+�                                    Byj��  �          @�p�@S33@���B�\�ffBN33@S33@��\��{�?�
BL
=                                    Byj��  
�          @��?�
=@�(����Ϳ�33B�?�
=@�33���H���B��{                                    Byj�0  
�          @�G�>��
@�Q�=���?�  B��H>��
@�  ��z��k�B��)                                    Byj��  �          @��H?8Q�@���<�>���B��q?8Q�@��׾����33B��                                    Byj�|  
�          @�33?8Q�@���>8Q�?���B���?8Q�@�G���\)�>�RB���                                    Byj�"  
�          @�=q?J=q@�  >�=q@7�B��H?J=q@�  �B�\��
B��H                                    Byj��  �          @���?h��@���>�ff@�
=B��?h��@�p�    =#�
B�                                    Byj�n  �          @���?��\@���?��@��RB��?��\@�p�=���?��B���                                    Byj�  "          @���?�  @�{>��@�z�B�33?�  @�ff�L�Ϳ��B�B�                                    Byk�  �          @��?^�R@�Q�?W
=A
=B�#�?^�R@��>�p�@}p�B�Q�                                    Byk`  T          @�Q�?G�@�(�?W
=A�B��=?G�@�{>�p�@s�
B��3                                    Byk*  �          @�=q?!G�@��R?^�RA��B��?!G�@�Q�>Ǯ@�  B�\                                    Byk8�  T          @�G�?E�@���?L��A��B��)?E�@�ff>��R@P  B�                                      BykGR  "          @���?8Q�@���?@  @�z�B���?8Q�@�{>�=q@5�B��                                    BykU�  �          @���?
=q@���?W
=Az�B��?
=q@�ff>�33@k�B�Ǯ                                    Bykd�  �          @���>�@���?n{A�HB�{>�@��>�ff@���B�.                                    ByksD  �          @��>L��@��?uA#�B��3>L��@�(�>�@��\B�                                    Byk��  �          @�
=��@�33?��\A,z�B�uý�@�p�?�@��B�p�                                    Byk��  
�          @�p�>W
=@�  ?��
A[33B�W
>W
=@�33?L��AQ�B�k�                                    Byk�6  
�          @�
=>��R@���?�=qAap�B�.>��R@�(�?W
=A=qB�L�                                    Byk��  "          @���>��R@��?�33A?�
B��>��R@��?&ff@���B���                                    Byk��  T          @���=#�
@��
?�\)Ad��B�{=#�
@�
=?^�RA�B��                                    Byk�(  �          @�G����
@��?���Aa��B�� ���
@��R?Y��Ap�B��                                     Byk��  �          @�Q�<�@��?�=qA��B�p�<�@��?��A7�B�u�                                    Byk�t  
�          @�ff>\@���@33A��B�aH>\@�{?�=qA���B���                                    Byk�  "          @�Q��Mp�@fff@�A�=qC��Mp�@qG�?�
=A�=qCs3                                    Byl�  
(          @��
����@o\)@EB��B�\)����@�Q�@.�RBG�Bߊ=                                    Bylf  T          @�ff��G�@�33@Dz�B
33B���G�@��
@+�A��HB��                                    Byl#  �          @��
��@z�H@;�B��B�8R��@�@#33A�p�B�.                                    Byl1�  �          @�  ��@i��@XQ�B (�B�G���@|��@AG�Bp�B��                                    Byl@X  
�          @�{��H@  @~�RBI\)C	(���H@'
=@p��B:{Cٚ                                    BylN�  T          @�ff�l(��#�
@h��B2p�C4.�l(�>���@hQ�B1�C.�)                                    Byl]�  T          @�=q�7
=?��
@���BZ�C =q�7
=?�
=@�p�BR�Cp�                                    ByllJ  �          @���7
=@\)@|��B=�RC�)�7
=@'
=@n{B/��C	��                                    Bylz�  "          @��\��R@Mp�@[�B!{B�\)��R@aG�@G
=Bp�B�B�                                    Byl��  �          @�  ���@\(�@N�RBffB������@o\)@8��B��B�ff                                    Byl�<  �          @����@U�@_\)B&G�B���@i��@I��B\)B�                                     Byl��  "          @���{@W�@P  B�B�W
�{@j�H@:=qBffB�Ǯ                                    Byl��  �          @�\)��p�@\��@S33B�\B���p�@p��@<��B{B�                                    Byl�.  T          @�녿�ff@w
=@QG�B��B�#׿�ff@��@8Q�B�RB֣�                                    Byl��  T          @�\)����@�{@,��A��
B�������@�{@  A��B��                                     Byl�z  �          @�(�?�=q@�?aG�A�B�=q?�=q@��>Ǯ@�ffB��R                                    Byl�   
�          @�33?��@�
=?�ffA�z�B�Ǯ?��@�(�?�=qAg�B��f                                    Byl��  T          @�33?���@���?�
=Az=qB��?���@�z�?uA%B���                                    Byml  �          @�p�?�Q�@���@�\A��
B�
=?�Q�@�\)?�A�B�#�                                    Bym  
�          @��R?˅@�33?�{A�  B�Ǯ?˅@�Q�?�\)AiG�B��H                                    Bym*�  
�          @�p�?���@��?���A�(�B��?���@���?�{A=�B�ff                                    Bym9^  
�          @���?�  @���?�\)A@��B��R?�  @��
?(�@�G�B�B�                                    BymH  T          @�{?E�@��?�Q�A�z�B�p�?E�@��?�
=Au�B�                                    BymV�  T          @�z�=���@�{@�A��B���=���@�(�?�{A�p�B��R                                    BymeP  �          @�?W
=@���@(Q�A��B��\?W
=@�z�@��A���B��\                                    Byms�  "          @�z�?��@��@A���B��f?��@�G�?˅A�33B�Ǯ                                    Bym��  
�          @�(�?E�@���@  A��HB���?E�@�Q�?�  A��RB��\                                    Bym�B  "          @�(�?8Q�@��H@p�A�ffB�\?8Q�@�G�?ٙ�A��B��R                                    Bym��  
�          @��?u@���@�RA�33B�.?u@�
=?�p�A���B�{                                    Bym��  
�          @��?
=@���@�A���B�  ?
=@��?�=qA��B���                                    Bym�4  �          @��H?p��@��@   A�{B�  ?p��@��H@ ��A��B�{                                    Bym��  
�          @��H?У�@���?˅A���B�u�?У�@�p�?�=qA<��B�p�                                    Bymڀ  �          @��?ٙ�@��R?��A<Q�B�#�?ٙ�@�G�?�@ÅB�                                    Bym�&  �          @��\?��
@��?�{AE�B���?��
@��\?
=@�{B�                                    Bym��  "          @�=q>u@��?�p�AV{B��>u@�Q�?.{@�B�Ǯ                                    Bynr  "          @��H=�G�@�{@33A��B�L�=�G�@���?��A��RB�k�                                    Byn  �          @�����@��@�A¸RB�
=��@�Q�?�z�A�(�B��                                    Byn#�  T          @�Q�!G�@���@QG�B��BŨ��!G�@�33@4z�B�B�\)                                    Byn2d  "          @�G����@��@P��B��BĊ=���@�@3�
B \)B�\)                                    BynA
  �          @��H��G�@z�H@eB)  B���G�@���@I��B��B��
                                    BynO�  
Z          @�녾��@c33@xQ�B=G�B�����@|��@^{B$B���                                    Byn^V  T          @��׿�G�@�Q�@/\)A��B�𤿡G�@���@��AͅB�{                                    Bynl�  "          @�  �G�@\)@S�
B\)B�#׿G�@��\@6ffB�B�z�                                    Byn{�  �          @���p��@dz�@i��B2��B�k��p��@|��@O\)B��BθR                                    Byn�H  T          @�\)��@j=q@Z=qB#�Bހ ��@�Q�@?\)Bp�B�{                                    Byn��  T          @�\)��\)@��
?���A��
B�(���\)@�  ?p��A&=qB�\                                    Byn��  �          @�녿5@^{@|(�B@(�B�33�5@x��@b�\B'\)B���                                    Byn�:  �          @�Q���
@��?p��A&{Bր ���
@��>Ǯ@�=qB�                                    Byn��  �          @�������@��?�Q�AQB�Ǯ����@���?#�
@�
=B�.                                    Bynӆ  �          @�=q��@��H?���A��HBԔ{��@��?�ffA7�BӨ�                                    Byn�,  T          @����\@��\@%�A�ffB�Ǯ�\@��H@�A��
BظR                                    Byn��  "          @����Y��@��H@=qAڣ�B��f�Y��@��\?��A��B��H                                    Byn�x  T          @�G��E�@�ff@�RAȸRBƳ3�E�@�?�Q�A��B��)                                    Byo  �          @�  ���@��H@�RA�{B�8R���@��H?�Q�A�ffB��                                    Byo�  
�          @�p����@��
@
�HA��HB�
=���@��H?��A���B���                                    Byo+j  T          @�����\@.�R@%�BC���\@@��@  A���B�u�                                    Byo:  
�          @��Ϳ�{@q�?��A�=qB�ff��{@}p�?���A��HB�Q�                                    ByoH�  �          @���(�@dz�?���A��B�33�(�@p  ?�33A�z�B��                                    ByoW\  Q          @�33��@C33@��B G�B�aH��@R�\?��A�G�B��                                    Byof  
�          @�ff�0��@0��@!G�B��C��0��@A�@(�A�RCW
                                    Byot�  T          @����#33@P  @z�A��HC (��#33@_\)?�
=A�Q�B�B�                                    Byo�N  �          @�  �=q@@��@0  B{C � �=q@S�
@Q�A�B�=q                                    Byo��  T          @�
=�:�H@��@1G�B�HC�
�:�H@-p�@{A�Q�C	+�                                    Byo��  T          @�(��Q�@P  ?�(�A��HB�(��Q�@]p�?���A�ffB΅                                    Byo�@  
�          @�?��@�p�>�ff@�ffB�33?��@�{��\)�c�
B�W
                                    Byo��  �          @�G��xQ�@��?���Ab=qB��
�xQ�@��H?
=@�Q�B�B�                                    Byǒ  T          @�z��p�@�(�?��Adz�B�\)��p�@��?#�
@��RB�W
                                    Byo�2  T          @�p���Q�@n{?��A�\)B��f��Q�@z�H?�
=A�z�B�                                    Byo��  �          @��R?��
@r�\@=qA���B���?��
@���?�Q�Aď\B��H                                    Byo�~  �          @�ff�u@x��@'
=B�\B��
�u@�@Q�A��
B��R                                    Byp$  �          @�zᾊ=q@vff@�B �B��f��=q@��
?�Q�Aə�B�aH                                    Byp�  T          @���G�@k�@
�HA���B�aH�G�@z�H?��HA��B�p�                                    Byp$p  "          @�\)���H@_\)@Q�A���B���H@p��?�
=A�Q�B��                                    Byp3  �          @�\)�"�\@8��@+�B��CaH�"�\@L(�@�
A�{C ��                                    BypA�  �          @�{��{@^�R@�RB �B�LͿ�{@p��@�\A��B�3                                    BypPb  T          @����R�\?���@.{B�RC^��R�\@G�@p�A���Ch�                                    Byp_  T          @���e?�@#�
BffC�R�e?�p�@�A�
=C�                                    Bypm�  �          @���|��?��H?��HA��
C"���|��?�?��A�C @                                     Byp|T  
�          @�{�a�?��R@*�HB�RC��a�?�@{A�
=CǮ                                    Byp��  �          @�33�E�@HQ�?��\A���C�\�E�@P��?^�RA.�HC\)                                    Byp��  T          @���Y��@0  ?��AV�RC\�Y��@7
=?0��A�C��                                    Byp�F  �          @����z=q@�>�p�@�G�C���z=q@p�=���?��RC��                                    Byp��  �          @�G��W�@2�\?@  A�CW
�W�@7�>Ǯ@���C��                                    BypŒ  �          @����Fff@
=?#�
A  C�q�Fff@�>�p�@��\C�                                    Byp�8  �          @��
� ��?�p�@��\Bc��C@ � ��@\)@tz�BO33C�                                    Byp��  T          @�z���?��H@qG�BK\)CQ���@�@_\)B7Q�C8R                                    Byp�  T          @�  ����?�(�@uB_Q�C
�H����@��@e�BJffC��                                    Byq *  T          @�Q���
@#�
@j=qBK33B����
@@��@S33B1(�B��H                                    Byq�  �          @�zῥ�@Q�@p��BX��B��ÿ��@6ff@Z�HB>33B�Ǯ                                    Byqv  �          @��Ϳ�ff@�@xQ�Bd�B�uÿ�ff@1G�@c33BH�
Bݏ\                                    Byq,  T          @��\�
=@  @�\)Bt��B���
=@2�\@x��BWG�B˸R                                    Byq:�  T          @�G����
@33@�G�B���B�\)���
@&ff@~�RBcp�B�p�                                    ByqIh  T          @�G�����?�G�@�\)B�B�G�����@�@��B33B�B�                                    ByqX  �          @�녿�?�ff@�{B��HC  ��?��@�\)ByQ�B��                                    Byqf�  T          @��H���?333@�z�B��3C� ���?��
@�  Bw=qCǮ                                    ByquZ  T          @�=q��  ?�=q@��\B�33CG���  ?�33@���Bo��C��                                    Byq�   T          @��ٙ�?�z�@�33Bv��CE�ٙ�?���@w�B`��C
                                    Byq��  �          @�z���
?���@���BsG�C#׿��
?�z�@s�
B^G�C��                                    Byq�L  �          @�  ��?�\)@�z�Br�C#���?�@|��Ba(�C��                                    Byq��  T          @�ff�*=q?�p�@b�\BE{C�H�*=q?���@S33B4(�CǮ                                    Byq��  T          @��!�@5�@'
=B
�C���!�@J=q@��A�ffC �3                                    Byq�>  �          @�����R@333@+�B�\C���R@H��@G�A���C J=                                    Byq��  T          @�(��\)?n{@j=qBWC��\)?�@_\)BJG�CB�                                    Byq�  �          @�{��Ϳz�H@��Bs�CK����;�p�@���B|\)C=�
                                    Byq�0  �          @�(��%��+�@��\Bc�CBu��%�����@�(�BgC6�                                    Byr�  �          @��
�%�#�
@���BbffCA��%��\)@�33BfC5�H                                    Byr|  �          @�{�8Q�s33@y��BP��CF8R�8Q�Ǯ@\)BW�
C;�=                                    Byr%"  �          @��\�H�þ�=q@`  B?��C8�R�H��>aG�@`��B@=qC0                                      Byr3�  T          @����!G�?z�H@n{BWC�{�!G�?��R@b�\BI�\Cp�                                    ByrBn  T          @��
�(�=u@��Bv�RC2Y��(�?#�
@�Q�Bq��C#�=                                    ByrQ  �          @�ff��33�^�R@���B|��CL����33��\)@�\)B�(�C<B�                                    Byr_�  T          @�{��G�����@���B��fC>�\��G�>�z�@���B�{C*�                                    Byrn`  �          @�(���
=>��@��\B��RC!Ǯ��
=?�=q@�
=B���C��                                    Byr}  �          @�
=�s�
�8Q�@1G�B�
C6���s�
>aG�@0��B�C0�q                                    Byr��  �          @�{�@  �J=q@^{B@�CB���@  ��\)@c33BF�RC9L�                                    Byr�R  T          @��R�^�R�L��@C33B!��C@�R�^�R��33@HQ�B'=qC9�=                                    Byr��  "          @���r�\��@1�B  C<���r�\�#�
@5�B33C6h�                                    Byr��  T          @��\��
=��
=@ ��A���CC�
��
=�^�R@
=qA���C?�                                    Byr�D  �          @����  ����?k�A+�CA���  �u?�=qAHQ�C?ff                                    Byr��  �          @�ff��33<��
?c�
A��C3����33>��?^�RA�C2\)                                    Byr�  �          @��R���?�?s33A)��C&�����?�ff?B�\Az�C%5�                                    Byr�6  T          @��
���?�\)���
��\)Cn���?��;����X��C��                                    Bys �  �          @�(��{�@7�����33C���{�@4z�
=q���C\)                                    Bys�  �          @�p��c33@Y��=L��?(�C:��c33@W�����=qC}q                                    Bys(  �          @�ff�R�\@j=q>��R@fffC���R�\@j�H�u�3�
C޸                                    Bys,�  �          @��A�@x��<�>�{B����A�@vff�\)��\)C 0�                                    Bys;t  "          @��H��G�@�{��{��
=B�{��G�@|(���p�����B�.                                    BysJ  T          @�녿�=q@�ff�����g�B�W
��=q@�
=��{��z�B�\                                    BysX�  "          @�z���@�녿Tz����B���@�zῼ(���33B�k�                                    Bysgf  "          @�{��(�@����  ��{B�{��(�@��\�	�����B�q                                    Bysv  �          @�p���ff@��Ϳ�  ��33B��Ϳ�ff@��\�=q���B�8R                                    Bys��  �          @������\@��H�ٙ�����B�uÿ��\@����ff��B�k�                                    Bys�X  �          @��H>�@���{���B��=>�@vff�6ff�G�B�G�                                    Bys��  �          @�G��z�@��R�\��33B���z�@�p��(����B���                                    Bys��  �          @�G��   @�p���  ���B�녿   @��H�=q����B��                                    Bys�J  �          @��ÿh��@�Q��G���(�B�{�h��@{��=q��Q�B�                                    Bys��  �          @����{@�Q��(����B��Ϳ�{@l������RB�                                    Bysܖ  �          @��ÿ�ff@�녿��
��
=B�\)��ff@~{�(���G�Bя\                                    Bys�<  "          @�Q쿂�\@�\)��33��z�B�33���\@xQ��#33��Bє{                                    Bys��  �          @��׿��@�����θRB�zῇ�@q��-p��
��B�8R                                    Byt�  T          @��׿��H@��������ffB�uÿ��H@hQ��3�
�(�B��                                    Byt.  T          @��ÿ��H@g��7
=�\)B�𤿚�H@G��Y���6  B�ff                                    Byt%�  �          @�p�����@�ff��G���  B�𤿨��@y���
�H��=qB�k�                                    Byt4z  �          @��
����@��R�����ffBה{����@|(���(����HB���                                    BytC   T          @�33�У�@_\)�p���B���У�@B�\�?\)�#p�B�\)                                    BytQ�  �          @�����
@vff����33B��)���
@`  �(���HB�B�                                    Byt`l  �          @�\)���@u�	�����BŸR���@\(��/\)��\Bǽq                                    Byto  
(          @�
=��\)@}p���
=�v�RB�\)��\)@n{������B���                                    Byt}�  �          @�33�%�@<���
�H��C5��%�@#33�'��p�CT{                                    Byt�^  
�          @��
�`��?�ff�z���p�C�{�`��?�33�%�	p�CQ�                                    Byt�  �          @�G��Y��?�\)�2�\�{C{�Y��?c�
�>{�!{C%\)                                    Byt��  �          @�=q�e�?��
�(Q��{C��e�?����8���G�C�R                                    Byt�P  �          @���Z�H>�=q�U��0��C/�=�Z�H��z��U��0�C8�H                                    Byt��  �          @����fff?W
=�<���G�C&��fff>����C33� p�C.                                    Byt՜  T          @���aG����S33�,��C5���aG��+��N�R�(Q�C>                                    Byt�B  �          @��R��H?�33�fff�F{C�)��H?�G��w��[G�Cc�                                    Byt��  �          @�
=��R?���o\)�U  C�{��R?!G��y���c  C%�H                                    Byu�  �          @��H�"�\?�Q��{��Z{C��"�\?   ���\�f\)C(��                                    Byu4  �          @����"�\?s33����`�HC��"�\>u����i��C.�                                    Byu�  �          @��ÿ��ý��
��z��C6z���ÿTz����B�CLz�                                    Byu-�  "          @�ff���׾�����{(�CDG����׿�z������C\�                                    Byu<&  �          @��\�Ǯ�#�
���B�C5z�Ǯ�Q���\){CO�f                                    ByuJ�  �          @��Ϳ�  �#�
��=q�C5^���  �Tz����ffCMn                                    ByuYr  �          @������>���33�zffC'����;�z�����|�C;aH                                    Byuh  !          @�����u��z��x�
C5�\���G���=q�qQ�CG��                                    Byuv�  �          @����1�?�(��Z�H�9=qCE�1�?���j�H�K\)C��                                    Byu�d  �          @����"�\?У��j=q�J�CJ=�"�\?u�x���\p�C@                                     Byu�
  �          @�G��,��@33�N{�)=qC���,��?�Q��e��AQ�C�f                                    Byu��  �          @��R�1G�@{�8���p�C
5��1G�?�
=�Q��0C0�                                    Byu�V  �          @�(��(Q�@g����
�O�B���(Q�@XQ��z���33B��3                                    Byu��  �          @�33�5@:=q@��B
=B�p��5@R�\?���A�(�B�\)                                    Byu΢  "          @���\(�@z�H@\)A陚B��Ϳ\(�@�Q�?�G�A�
=B��f                                    Byu�H  T          @�z�z�H@^{@$z�B��B�uÿz�H@w�?�33A�{B�Q�                                    Byu��  �          @�z��G�@L(�@4z�B�RB�Ǯ��G�@i��@��A�B�\                                    Byu��  �          @��
���@�G�?�@�G�B�����@�=q�.{��RB�p�                                    Byv	:  �          @��\��@vff?��A]p�B��ÿ�@~�R>�{@��RB�z�                                    Byv�  T          @��H��{@�p��������B�#׿�{@��׿�33�hz�B垸                                    Byv&�  "          @�(��\)@ ��@@  B'
=C���\)@AG�@\)B\)B��                                    Byv5,  �          @�ff�)��@1G�@"�\B{C�R�)��@L(�?��RA���C�R                                    ByvC�  �          @�  ��\)@�p�?n{A9�B�z῏\)@���=���?��B��f                                    ByvRx  T          @�����R@���?�p�Atz�B�����R@�>�@��RB��                                    Byva  T          @�ff�.{@qG�>�z�@g�B��\�.{@p�׾����G�B��R                                    Byvo�  �          @�ff��
@�녾�  �@��B�Q���
@|(���  �D��B�Ǯ                                    Byv~j  �          @�{��@�G���ff�Qp�B�\)��@qG����
���B�\                                    Byv�  T          @����(��@s�
=u?E�B�aH�(��@p  �&ff�G�B�.                                    Byv��  "          @�z��p�@tz�>u@A�B��=�p�@r�\����=qB��)                                    Byv�\  �          @�
=���H@��
?��\Az�RB��f���H@���>�(�@�{B��
                                    Byv�  �          @�p����@mp�?�Q�A�=qB�R���@}p�?uA>{B�ff                                    ByvǨ  �          @�Q쿅�@�{?�ffA�
=B�𤿅�@�ff?xQ�A?�
B�Q�                                    Byv�N  �          @�\)��G�@�33>���@qG�B�녾�G�@��\�z����B���                                    Byv��  T          @�\)��  @�>#�
?�(�B�8R��  @�(��=p��z�B�G�                                    Byv�  
�          @�\)>��@�G�?�R@��B�ff>��@�=q����S33B�u�                                    Byw@  �          @��ý�\)@�ff�������B�Ǯ��\)@�33�p���ffB��                                    Byw�  T          @�  ��\@�ff���
���
B�B���\@�33�}p��Ap�B��=                                    Byw�  
�          @��׿\)@��R�8Q���RB�z�\)@��\����UG�B��
                                    Byw.2  �          @��׿\(�@��=u?=p�B��Ϳ\(�@��H�Y���$��B�#�                                    Byw<�  
Z          @�����\@�=q?8Q�A��B�8R���\@��
�8Q���B��                                    BywK~  
Z          @�녿\)@�p�?E�A�B��{�\)@���#�
���B�ff                                    BywZ$  
�          @����z�@��?���AN�HB�\)�z�@�\)>\)?�Q�B�                                      Bywh�  �          @�33��@�Q�?(��@��B�����@�����z��X��B��=                                    Bywwp  
Z          @�=q��{@�  ��Q�����B����{@��H����z�RB�ff                                    Byw�  "          @����G�@���>\@�p�B��G�@�(��
=q���
B�{                                    Byw��  
�          @��\��{@��?B�\AG�B�\��{@�p��8Q��Bνq                                    Byw�b  
*          @����ff@��
������B�����ff@�Q쿁G��=�B��f                                    Byw�  
(          @�(��z�H@�  �\)��z�B�LͿz�H@��
����O�B��f                                    Byw��  T          @��R���@�  ��G��mp�B�W
���@����(���
=B�8R                                    Byw�T  "          @�
=��Q�@�p���R��ffB�\)��Q�@xQ��E���RB��3                                    Byw��  �          @�G���\)@�p�� ����=qB��ý�\)@j�H�4z���
B�8R                                    Byw�  �          @�  >�  @�(�������B�z�>�  @z�H�(�����B��R                                    Byw�F  
�          @���8Q�@�(���G���  B��3�8Q�@~�R����  B�.                                    Byx	�  
�          @��;�@��H��(����B�Q��@�p������33B�u�                                    Byx�  �          @��H=��
@�Q��
=���B�=��
@��\������B��                                    Byx'8  
Z          @����E?�p�?�\)A�G�Cs3�E?�?J=qAL  C��                                    Byx5�  �          @�=q���ÿE�?���A��C>�{���þ�33@�
A�Q�C8�q                                    ByxD�  �          @����{����@
�HA�{CC#��{��
=@A��
C<�\                                    ByxS*  �          @��
�Q�>\@)��B��C-n�Q�?k�@!G�B��C$T{                                    Byxa�  R          @����[���@A�B$ffC<�q�[�=�@Dz�B'G�C2�                                    Byxpv  V          @�Q��Z=q?k�@333B��C$��Z=q?�p�@"�\B	��C�                                     Byx             @�  �u���@
�HA�=qC<� �u�����@\)A���C5�\                                    Byx��  	�          @����s33���@��A���CC�H�s33�z�@�A��C<�H                                    Byx�h  	�          @���u�#�
@p�A�G�C4��u>�@
=qA陚C-+�                                    Byx�  
�          @�Q��c33>�=q@#33B��C/�3�c33?L��@(�BG�C'E                                    Byx��  
(          @�  �G
=��p�@5B
=CMz��G
=�W
=@EB/=qCC.                                    Byx�Z  "          @���u��0��@A��C>8R�u��8Q�@�BffC6��                                    Byx�   �          @�z�������?��HA�(�C8���=���?�p�A���C2�                                    Byx�  
�          @��
���>�p�?��HAɅC.�f���?G�?�A��
C)Q�                                    Byx�L  "          @�33���?��\?��A�p�C&:����?���?���A�(�C"
=                                    Byy�  "          @�33�\)?�=q?�{A���C!���\)?ٙ�?��
A�G�C��                                    Byy�  �          @��\�j=q?�  @�
A�{C��j=q?�(�?�
=A�
=C�q                                    Byy >  
�          @�z���  ?�
=?.{A33Cn��  ?��>���@l(�C0�                                    Byy.�  "          @�G����?ٙ�?��@��
C�q���?��>aG�@.{C��                                    Byy=�  T          @�  ����?��R?Y��A.�RC @ ����?��?   @�{Cp�                                    ByyL0  
�          @�=q���?���>8Q�@��C#)���?�=q��G�����C#                                    ByyZ�  T          @��\����?�(�<��
>��RC$�\����?�Q�u�C�
C$ٚ                                    Byyi|  "          @������\?��R>k�@9��C#�q���\?�G��L�Ϳ(��C#�                                     Byyx"  �          @�=q��(�?��<��
>L��C#�)��(�?�G���=q�X��C#��                                    Byy��  �          @�=q��p�?�=u?G�C����p�?�\��{����C��                                    Byy�n  �          @�=q���\@�>�{@���C}q���\@�
����  C33                                    Byy�  "          @��H�s�
@�>�G�@�Q�C���s�
@z���\Cp�                                    Byy��  �          @����p��@�
>.{@p�C@ �p��@33��������Cn                                    Byy�`  �          @�Q��q�?�>\@�
=C\�q�?�׽#�
�
=qC�{                                    Byy�  �          @����o\)?�?aG�A?
=C���o\)?��R>�G�@�\)C��                                    Byyެ  �          @�Q��c33@
=q>�33@��C�f�c33@������\CaH                                    Byy�R             @�G��5@��@�A�C��5@%�?��A��\C	�                                    Byy��  
*          @��H�P  @"�\?�=qAjffC��P  @.{>��@˅C)                                    Byz
�  "          @�{�a�@'
=?!G�A�HC���a�@+�<��
>uC�                                    ByzD  �          @����`��@(��?�{Ab{C�`��@4z�>��@�
=C&f                                    Byz'�  �          @�\)�E�@,��?�A�
=C
�=�E�@B�\?��Al��C\)                                    Byz6�  �          @����L��@.{?�33A�p�C���L��@A�?z�HAH  C�{                                    ByzE6  T          @����K�@+�?��
A��
C޸�K�@AG�?�\)Ad��CxR                                    ByzS�  "          @����1G�@:�H@�\A��HCxR�1G�@Tz�?��A��\C�H                                    Byzb�  T          @�  �*=q@<��@�Aܣ�C\�*=q@Vff?�=qA�Q�C u�                                    Byzq(  T          @���p�@E?���Aљ�C ���p�@]p�?�Av�\B��f                                    Byz�  "          @�p��%@E?��
A���C���%@Z�H?�G�AT(�B�G�                                    Byz�t  
�          @�(����@N{?�z�A���B�z����@aG�?\(�A5B���                                    Byz�  �          @��
��@Vff?˅A��
B�z���@hQ�?B�\A!�B�=q                                    Byz��  �          @��
�{@L��?�{A��
B�\)�{@_\)?O\)A+�
B���                                    Byz�f  
�          @��
�
=@Z�H?��HA�z�B�B��
=@n�R?Y��A4(�B��                                    Byz�  "          @�(��$z�@E?���A���C���$z�@XQ�?Q�A.�RB�aH                                    Byzײ  T          @�33�(��@Dz�?��A��HC���(��@Vff?E�A#33C :�                                    Byz�X  T          @���(Q�@<��?\A�  C��(Q�@N�R?E�A'�C:�                                    Byz��  T          @���(Q�@:=q?�ffA��\C#��(Q�@L(�?O\)A0��Cz�                                    By{�  �          @��R�Z�H@!�?���A�{Cu��Z�H@2�\?:�HA33C�                                    By{J  
�          @����]p�@'�?��HAxz�C�
�]p�@5?
=q@��
C��                                    By{ �  
X          @��
�o\)@�
?�z�A���CJ=�o\)@%�?O\)A"ffC\)                                    By{/�  
�          @�=q�c33@?�33A�p�C�)�c33@*�H?��
AQ�C\                                    By{><  
Z          @�33�w�?�?�p�A��RC��w�@
�H?��RAz{C��                                    By{L�  �          @�33�xQ�?�
=?��RA�G�C�q�xQ�?�\)?�=qA��HC:�                                    By{[�  T          @�(��~�R?�=q?���A��C!xR�~�R?�\?���A���C                                      By{j.  �          @�(��~{?�\)@33A��HC$:��~{?˅?��HA�{C33                                    By{x�  �          @�����Q�?���?У�A��C8R��Q�?���?�Q�An�\C                                    By{�z  V          @�(���
=?�G�?���Aa�C L���
=?޸R?8Q�A\)C�\                                    By{�   "          @�
=��(�?��\?��AQG�C#�)��(�?��R?5AQ�C!5�                                    By{��  �          @�
=��p�?�R?�G�A��C+����p�?z�H?��A���C'��                                    By{�l  T          @�
=���\?�R?�33A�p�C+�
���\?�G�?���A�  C&ٚ                                    By{�  T          @�
=����?333?�=qA�C*������?��?���A�\)C%�                                    By{и  
�          @�{�]p�@	��@�A�C��]p�@(Q�?�=qA���C�                                    By{�^  
�          @����8��@9��@�A���C�H�8��@U?���A���C�)                                    By{�  T          @�33��33>�?��HA�=qC.  ��33?W
=?��Aw\)C)��                                    By{��             @�����Q�>��?���A���C.� ��Q�?L��?�A�\)C)��                                    By|P  
�          @��\��=q>�=q?ǮA�Q�C0����=q?(��?�Q�A�ffC+��                                    By|�  �          @���
=<#�
?���A�=qC3�)��
=>�Q�?�=qA{33C/�H                                    By|(�  "          @�=q���
���?�{Ay��C7\���
=�Q�?���A}��C2��                                    By|7B  T          @�\)��  ��?��A�p�C9�q��  ��G�?��HA���C5O\                                    By|E�  �          @����p��G�?��AO�C=z���p���?�  Ao33C9��                                    By|T�  �          @��\���\�\(�?��RAo33C>�����\��\?�A�G�C:aH                                    By|c4  	�          @�p�����fff?��As�C>�����
=q?�p�A�  C:�{                                    By|q�  
Z          @����z�0��?�33A]�C<p���z᾽p�?��Ax(�C8�=                                    By|��  �          @��������p�?�G�AvffC8������#�
?��A���C4p�                                    By|�&  �          @�=q���<�?�Q�Ap��C3�����>���?��Af�HC/��                                    By|��  �          @�����
=?z�H?�ffA�G�C&ٚ��
=?��?�  AK�C#                                    By|�r  �          @��H���\?\(�?���Ar�\C(�����\?��?n{A<(�C%B�                                    By|�  	�          @�Q���?�z�?���AX  C����@Q�?�@��C
=                                    By|ɾ  T          @�(��z�H?���?�Aw�
C5��z�H?���?8Q�A33C�                                    By|�d  "          @�G����?(�?�Q�A��C+�����?fff?}p�AUC'��                                    By|�
  	�          @��~�R>#�
?��A�{C1���~�R?   ?�(�A��C,�\                                    By|��  �          @z=q�\(�?���?���A��RC"0��\(�?�z�?z�HAm�C�f                                    By}V  T          @p���@  ?��?s33As
=C@ �@  ?�(�>�G�@���C�                                    By}�  "          @l(����@*=q?B�\A?
=C  ���@1G�<�>���C��                                    By}!�  �          @����0��@>{?L��A1��Cٚ�0��@E����
���RC޸                                    By}0H  
�          @����$z�@Vff?z�@�=qB���$z�@XQ쾨�����B�\)                                    By}>�  "          @���0��@A�?&ffA(�Cff�0��@E�8Q�� ��C�\                                    By}M�  T          @u��
=@L(�<#�
=�G�B�B���
=@E��Y���`Q�B��
                                    By}\:  �          @w�@QG�?�Q�   ��Q�A��H@QG�?�p���  �rffA�
=                                    By}j�  �          @��H@_\)@33��G����A�@_\)?�{�xQ��]�A�=q                                    By}y�  �          @��@:=q@2�\>��@(�B/(�@:=q@.�R�(��	B,��                                    By}�,  T          @��@��@]p�>8Q�@Q�B]\)@��@XQ�E��(  BZ��                                    By}��  "          @�(�@9��@6ff=���?��B2
=@9��@1G��.{�=qB.�R                                    By}�x  �          @�Q�@�
@=p��0���&�RBP
=@�
@*=q��(���
=BD{                                    By}�  
�          @��
@)��@5���  �g\)B;@)��@�Ϳ�p���G�B+G�                                    By}��  
�          @xQ�@p�@�ÿ޸R��z�B${@p�?˅�����\B�                                    By}�j             @u?�33@Q��=q��Q�B�Q�?�33?�{�
�H�3  Bh                                    By}�  $          @O\)�.{?�ff?z�A(��C\)�.{?�33>\)@��C�=                                    By}�  "          @W��E�?}p��   �p�C"=q�E�?O\)�@  �Up�C%B�                                    By}�\  �          @Vff�Fff?�33>���@�p�C�{�Fff?�Q����C�R                                    By~  
�          @U�@��?�z�<��
>ǮC޸�@��?�{��p����
C��                                    By~�  �          @W
=�7
=?��H��  ���
C\�7
=?��ÿ8Q��F�HC!H                                    By~)N  
�          @Q��G
=?G�>�  @��
C%�q�G
=?O\)<��
>�33C%W
                                    By~7�  T          @?\)�<��>L��>�=q@�\)C0!H�<��>�=q>L��@|��C.�{                                    By~F�  T          @H���7
=�s33�333�P  CFc��7
=��{������CI(�                                    By~U@  
Z          @aG��O\)=��ͿB�\�T��C2J=�O\)���B�\�S�C6&f                                    By~c�  �          @\���-p�@��\)��\)CaH�-p�?�
=�\(��g\)C�)                                    By~r�             @o\)���@1녽��ͿǮC����@(�ÿaG��]G�C=q                                    By~�2  
�          @�(����@tz�B�\�%�B�����@g
=��  ��B�{                                    By~��  
Z          @j�H�#�
@{����z�C{�#�
@\)�����(�C
��                                    By~�~  
(          @W
=������@�\B2
=C6Y���?\)@{B+�\C&��                                    By~�$  T          @W
=�!녾B�\@�B"z�C8ff�!�>�(�@��Bp�C*T{                                    By~��  T          @P  �{>W
=?�B�C/&f�{?@  ?��B
�HC#�                                    By~�p  T          @qG���Q�?�ff@-p�B>z�C@ ��Q�@   @�B
z�B�G�                                    By~�  �          @xQ����@@��?�=qA�  B��ÿ���@Y��?c�
AU�B�G�                                    By~�  �          @�Q�\)@hQ�?�33A���BŊ=�\)@xQ�>���@��
B�p�                                    By~�b  
�          @���=#�
@��?\(�A;
=B��=#�
@�\)��p����B��                                    By  �          @��׾�@�(�?0��A=qB��{��@������\)B��                                     By�  
Z          @�{�\(�@XQ�?���A߮B�ff�\(�@q�?L��A:�\B͏\                                    By"T  �          @��׿333@e�@�A�B�B��333@�G�?�  AZ�RB�                                    By0�  "          @�  �\)@��?���AjffB����\)@�ff�.{�(�B��                                     By?�  T          @�(���\)@��\?   @�  B��R��\)@��ÿY���)�B��q                                    ByNF  R          @�(���G�@�\)?   @ȣ�B�k���G�@��Q��#�
Bͮ                                    By\�  �          @���u@��
>\@�ffB��f�u@��ÿz�H�C�
B�                                      Byk�  T          @��R�Ǯ@�p�=�?\B����Ǯ@�  ��  �v�RB���                                    Byz8  �          @�
=�!G�@x��?�@�G�B�Ǯ�!G�@xQ�!G���G�B���                                    By��  �          @�{�@�Q�>��R@s33B�z��@{��aG��/
=B�{                                    By��  �          @�\)�G�@��Ϳ��
�O\)B�  �G�@y������Q�B�z�                                    By�*  T          @�ff=�G�@��H���H���\B�\=�G�@h���:�H��
B�z�                                    By��  "          @�z�?B�\@�G���
=��G�B���?B�\@k��(���(�B��                                    By�v  
(          @�{?h��@�G���33��
=B�\?h��@g
=�6ff��B��R                                    By�  T          @�Q�>�G�@�녿��R�r�\B�>�G�@~�R�"�\�\)B�u�                                    By��  
�          @�33?���@|(��u�c�
B��?���@p  ��p���\)B�k�                                    By�h  T          @�33?aG�@��׿333�{B�?aG�@w���Q��иRB�ff                                    By�  T          @�
=?0��@}p�����z�B�L�?0��@N�R�:=q�%�\B�                                      By��  "          @�\)?G�@}p���Q���p�B��{?G�@Z�H�z���HB�(�                                    By�Z  
�          @�녾�(�@��Ϳ.{�
=B��)��(�@p�׿����ffB��                                    By�*   �          @�{?�{@\(��G�����B��?�{@%�L���933Bhp�                                    By�8�  �          @�\)?��
@\������z�B��?��
@$z��Tz��@  Bl�R                                    By�GL  
�          @�p�?��R@7
=�E�2�B�.?��R?�ff�s�
�p�
B]�                                    By�U�  
\          @��H?   @�(�����=qB�8R?   @W
=�E��(��B�{                                    By�d�  "          @���>B�\@z�H�p����B�� >B�\@Dz��S33�<{B���                                    By�s>  �          @���?L��@Q��8Q��!B��?L��@  �o\)�e�B�                                      By���  �          @���?u@8���Vff�>��B�=q?u?޸R��=q���Bt(�                                    By���  �          @��?�@�R�O\)�7�
BQ
=?�?�\)�u�i�
B=q                                    By��0  
Z          @��
?���@B�\�HQ��+�B�\?���?�Q��z=q�h�BT�                                    By���  �          @�=q?��@Vff��p���ffB�W
?��@/\)�(��$p�B�=q                                    By��|  "          @��R��33@~�R?�
=A�{B��ᾳ33@�(�?(�@�{B�{                                    By��"  "          @����^�R@�  ?��A�
=B�B��^�R@�ff����G�B�8R                                    By���  �          @�ff��z�@���?�ffA��B�\)��z�@�  >�G�@��B�                                      By��n  "          @�33��G�@���?�  A|z�B�\)��G�@�\)�\)��\B�\)                                    By��  �          @����z�@��R?���A�33B��ᾔz�@��\>�ff@��B�W
                                    By��  T          @�p��z�H@�z�?�(�A�=qBΙ��z�H@�z�=�G�?�z�B��                                    By�`  "          @�=q��@�33?���AfffB�  ��@�\)��\)�c�
B\                                    By�#  
�          @���ff@P��@#33B(�B߀ ��ff@x��?�z�A��
B���                                    By�1�  T          @��ÿ�ff@��@c33BOQ�C\��ff@H��@-p�BG�B�{                                    By�@R  T          @�ff���?�@l��BP�\C
)���@>�R@:�HB��B�aH                                    By�N�  
�          @�����@8��@HQ�B&(�B�G���@n{@z�A�B�B�                                    By�]�  T          @�33���@6ff@O\)B(33B�L����@n{@(�A֣�B��                                    By�lD  
&          @��(�@R�\@:=qBp�B�#��(�@�G�?�p�A��RB���                                    By�z�  
�          @�33�   @h��@!�A��RB�=�   @�  ?�  An�HB�\)                                    By���  �          @�\)�!�@8Q�@%B=qCL��!�@c33?�ffA��\B��                                    By��6  T          @��ÿ�Q�@Fff@L��B,\)B�𤿸Q�@|��@�
A�(�B�#�                                    By���  
�          @�Q��
�H@s33?���A���B�p��
�H@�{?��@�B�R                                    By���             @���)��@^�R?��A��B����)��@y��?+�A=qB��                                    By��(  $          @����>�R@4z�@�RA�\)C���>�R@]p�?��HA��C                                    By���  �          @��\�P  ?��
@?\)B�CO\�P  @(��@G�A㙚C�                                    By��t  �          @��\�Q�?���@X��B2�C!�R�Q�@@7�B��Ck�                                    By��  
�          @��R�S33>�
=@j�HB?z�C,�S33?���@VffB*  C��                                    By���  �          @����U�>Ǯ@[�B6C-G��U�?�(�@G�B"��C#�                                    By�f  
�          @��H�:=q?��@P  B>z�C)L��:=q?Ǯ@9��B%  C                                    By�  �          @�Q쿷
=@HQ�@0��B�B����
=@w
=?�{A���Bܽq                                    By�*�  
�          @�(����@l(�@z�A��
B�Q쿑�@��?�  ANffB�{                                    By�9X  
�          @�녾�z�@��?(�@��\B��׾�z�@��
�Y���/�B��R                                    By�G�  
(          @�\)?�=q@o\)@\)A�\)B���?�=q@�Q�?h��A:�HB��{                                    By�V�  �          @��?���@���?��RAz=qB�ff?���@�
=�aG��.{B�u�                                    By�eJ  
�          @��?z�H@��>L��@=qB�
=?z�H@�������HB�{                                    By�s�  �          @��
?���@�ff�+���(�B�=q?���@�
=�
=q����B�z�                                    By���  �          @��?&ff@�33=�?�Q�B��q?&ff@��
���H��33B�                                    By��<  �          @��?�z�@�  �����  B�� ?�z�@�33��
=���
B��                                    By���  �          @���?B�\@�  �\(��"�HB�?B�\@��R����  B�u�                                    By���  
�          @���?�\@���p��jffB���?�\@����,�����B��{                                    By��.  "          @�\)?��@����   ��(�B��=?��@y�������33B�                                    By���  �          @��\?���@��R��33�iG�B��?���@fff�{�{B���                                    By��z  �          @�(�?z�H@���n{�:{B�?z�H@w
=������B��{                                    By��   
�          @�  ?5@��׿��H�yG�B�.?5@hQ��#33�
33B���                                    By���  
Z          @���?�ff@~{��G���B�Q�?�ff@J=q�>{� �RB�\                                    By�l  R          @�33@
=@b�\�#33��{Bl@
=@   �e��>ffBG{                                    By�  
�          @�?   @�  �c�
�2{B�\)?   @|��������B���                                    By�#�  T          @�(�?Y��@�\)�ٙ����B�u�?Y��@j=q�E��RB�Ǯ                                    By�2^  
Z          @��>��@����	�����HB���>��@Tz��\���8
=B��                                    By�A  �          @��?��@w�������p�B���?��@G��1G��$�B�                                    By�O�  T          @��?��R@�\)�\)��\B�(�?��R@|(��������B��=                                    By�^P  �          @�(�?�ff@�G��u�Ap�B�(�?�ff@n{����B�\)                                    By�l�  T          @�Q�?�  @��׿�z���{B�=q?�  @c33�0���33B�(�                                    By�{�  "          @�  ?�Q�@����Q��i�B��?�Q�@^�R�   � p�Bsff                                    By��B  
�          @��@�@��H���H����Bw�H@�@W
=�/\)�	BdG�                                    By���  "          @�
=@(�@z=q��\)����Bh(�@(�@C33�Dz���HBMz�                                    By���  �          @�33@,��@g
=��p����BT�@,��@4z��6ff�z�B8�H                                    By��4  �          @�\)@Fff@TzῨ�����B;��@Fff@+��
=���B#G�                                    By���  "          @�p�@8Q�@G
=���¸RB<�
@8Q�@��6ff��\Bff                                    By�Ӏ  
�          @�\)@&ff@W
=��\����BP��@&ff@{�C�
�!�B.
=                                    By��&  
�          @���@!G�@Z=q�  ��{BV=q@!G�@(��QG��,G�B0z�                                    By���  
�          @�@-p�@R�\��
=��ffBJ(�@-p�@(��;���B({                                    By��r  	�          @���@5@'��ff���HB+  @5?�z��G
=�-z�A�
=                                    By�  
Z          @��@
=@J=q������(�BT�@
=@ff�2�\�B3ff                                    By��  �          @�\)?fff@|�;����
B�aH?fff@j=q��(����
B�z�                                    By�+d  
�          @��H?�
=@,���)���\)BZ  ?�
=?У��Z=q�V  B!
=                                    By�:
  �          @�G�?�33@u������Q�B��H?�33@J=q�!G���B�33                                    By�H�  T          @���?���@x�ÿ��
���B���?���@N{�"�\��B�(�                                    By�WV  
�          @�=q?Y��@|(��.{�=qB��{?Y��@h�ÿ�G���Q�B���                                    By�e�  �          @���?�  @�  ?p��AM��B��?�  @��\�   ��
=B�p�                                    By�t�  �          @�  ?���@���?G�A(z�B�33?���@����(����
B�W
                                    By��H  T          @�?��@��?:�HA!B�=q?��@�=q�8Q��{B�=q                                    By���  T          @��?^�R@�33>���@u�B�k�?^�R@���(���{B��=                                    By���  "          @���?��@�z�=#�
?�B�{?��@�(���p����B�k�                                    By��:  T          @�z�?�z�@�Q�����Q�B���?�z�@mp����
����B��q                                    By���  T          @��R@ff@c�
�+���HBn
=@ff@G
=��{�ӅB_��                                    By�̆  
�          @��?�p�@fff��33��=qBt�H?�p�@P�׿�=q��  Bj�
                                    By��,  T          @�Q�?�(�@n{�aG��C�
Bxz�?�(�@Z=q��G����Bp                                      By���  
�          @��@��@\(�>L��@-p�BZ{@��@R�\��G��]�BU\)                                    By��x  �          @���@`  ?��?k�AQG�A�@`  @�>W
=@:�HA��                                    By�  
�          @�G�@���?�>�
=@��A���@���?�(���G�����A�33                                    By��  
�          @�G�@tz�?�Q�=�Q�?�
=Aׅ@tz�?��Ϳ��� Q�AΣ�                                    By�$j  
�          @�Q�@o\)?��B�\�%�A��H@o\)?�p��Y���:ffA�(�                                    By�3             @��R@=q@S33?B�\A)p�BWff@=q@Vff��G���z�BYQ�                                    By�A�  
�          @���@G�@P  ?�A�33Bh��@G�@b�\>B�\@'�BqQ�                                    By�P\  �          @}p�>.{@>{@
�HB\)B��>.{@b�\?z�HAw�B��\                                    By�_  �          @���>W
=@L(�@�B	z�B��=>W
=@p  ?k�A[�B���                                    By�m�  
(          @�z�?=p�@�@QG�BS\)B�p�?=p�@W�@{B�B�\)                                    By�|N  �          @�z�?+�@,(�@?\)B<\)B�{?+�@e�?�A�ffB���                                    By���  �          @�(�?��H@33@A�B@
=Bf��?��H@N�R@ ��A�
=B�p�                                    By���  
�          @�  @�?���@N�RBG=qB'�@�@6ff@�B��BY                                    By��@  �          @�=q?���?\@j=qBpp�B9p�?���@.{@8Q�B,  Bw��                                    By���  
�          @�G���
=@<��@�RB(�B�ff��
=@c33?��
AvffBؙ�                                    By�Ō  "          @|���#33@/\)���Ϳ��C�q�#33@!G���������CW
                                    By��2  T          @���K�@#�
�����Q�C@ �K�@{������33C)                                    By���  T          @�z��e@�>k�@HQ�C�{�e?�(���\��z�C:�                                    By��~  T          @��\�L��@��?+�A��C0��L��@{�k��O\)CB�                                    By� $  �          @�33�5@<��>�\)@}p�C��5@6ff�J=q�2=qC�)                                    By��  �          @��H�7
=@+�?��An{C��7
=@7�<#�
>#�
C�                                    By�p  �          @���G
=?���@#�
Bz�C!��G
=?�z�@ ��A�  Ck�                                    By�,  
Z          @�  �C33?��@.{Bz�C�
�C33@�\@Q�A��HCB�                                    By�:�  
�          @�p��#�
?�33@%B��Ch��#�
@-p�?�p�Aƣ�Cp�                                    By�Ib  �          @�녿�{@XQ�?�33A��
B�𤿮{@j=q=��
?�p�B��f                                    By�X  
�          @��H��(�@0��@(��B �B�𤿼(�@a�?��HA�G�B�#�                                    By�f�  
\          @��Ϳ�  @��@FffBCffB���  @W
=@G�A��B��
                                    By�uT  T          @��ÿ�=q@7�@3�
B&p�B��)��=q@l��?�=qA�\)B۸R                                    By���  	�          @����z�@c�
    �#�
B���z�@Tzῥ����B��)                                    By���  �          @����{@j�H>��@�
B�(��{@^�R��Q�����B���                                    By��F  
�          @�  �33@l�ͼ��
���
B��)�33@\(���\)��
=B�z�                                    By���  T          @����@Z=q�������B����@C�
��ff��(�B�z�                                    By���  T          @�����\@Z=q���H��  B�����\@@  �ٙ���B���                                    By��8  
�          @����Q�@{���z�����B�\)��Q�@c33���H��(�B�B�                                    By���  
�          @�녾�Q�@�(������nffB��f��Q�@\(��"�\�  B��H                                    By��  �          @��\�O\)?�R?�  A��C)+��O\)?�(�?�33A��Cp�                                    By��*             @���I����Q�@P��B7�RC5���I��?�{@C�
B*{C ��                                    By��  V          @�\)�c�
?(��@�HB  C)n�c�
?�G�@ ��AۅC{                                    By�v  
�          @�{�s�
?�G�?�
=AЏ\C%(��s�
?�33?�A�\)C�)                                    By�%  "          @�
=�Z=q?��@!�B
�C:��Z=q@Q�?�\)A�  C��                                    By�3�  
�          @�  �O\)?��@*�HB�\C���O\)@=q?�A��HCW
                                    By�Bh  �          @���[�?�=q@�HA��C�3�[�@%?�=qA���C�                                    By�Q  �          @���E�?��R@/\)B��C0��E�@7
=?�A�z�C	.                                    By�_�  	`          @�=q�=p�@�
@0��B��C)�=p�@;�?�ffA���C8R                                    By�nZ  	�          @����:�H?�\@<��B#{C���:�H@.�R@�A�  C�                                    By�}   T          @����=p�?��H@FffB,�C���=p�@   @A��C��                                    By���  �          @�{�>{?��R@;�B)p�Ch��>{@�R@��A��HC�                                    By��L  
�          @���?L��@`  BWz�C!�
��@�\@=p�B+��C��                                    By���  �          @����aG�?�
=@�\A�G�Ch��aG�@=q?�  A��HC�=                                    By���  
�          @�Q��I��@   @!�Bz�C�=�I��@333?���A�z�C
\)                                    By��>  T          @��H�QG�@@p�BffCaH�QG�@7
=?�  A��RC
�
                                    By���  �          @��
�AG�@G�@%B	C�q�AG�@E�?ǮA���Cp�                                    By��  T          @�
=�O\)?�ff@9��B
=C�R�O\)@0  @G�A�
=C��                                    By��0  �          @��b�\?�@ ��B�
C�q�b�\@\)?��HA�=qC��                                    By� �  
�          @��R�h��?�33@Q�A�(�C�)�h��@�H?˅A��
C^�                                    By�|  "          @��R�^{?�p�@��A���C@ �^{@0��?��
A���C��                                    By�"  �          @�ff����?�ff?�Q�A�=qC".����?�
=?��A�Q�Cu�                                    By�,�  
�          @�{�b�\?��H@�
A��
C�b�\@+�?�z�A��\C��                                    By�;n  �          @�
=�i��@z�@	��AمCaH�i��@.�R?�(�Ao�C+�                                    By�J  
�          @�\)�a�?�33@!G�B C�3�a�@-p�?�\)A�Q�Cz�                                    By�X�  �          @�  �QG�?�Q�@5�B�RC=q�QG�@7�?��A�{C
��                                    By�g`  
Z          @���5�@z�@7
=B{C���5�@N�R?��
A��
C@                                     By�v  "          @���H��@��@&ffB	�C�f�H��@>{?���A��\C��                                    By���  "          @��H�.{@z�@7
=B�\Cz��.{@N�R?�\A��C{                                    By��R  �          @��\�7�@
=@7
=B��C� �7�@A�?���A��CxR                                    By���  T          @�z��E�@ff@#33B�C���E�@H��?�(�A�=qCxR                                    By���  
�          @����H��@z�@   B�C���H��@E?�Q�A�ffC}q                                    By��D  �          @���J�H?�z�@5�Bz�C��J�H@5?��A���C
#�                                    By���  �          @��
�I��@�@*�HB33C}q�I��@<(�?�
=A���C�                                    By�ܐ  
Z          @�33�A�@33@$z�B�C�
�A�@Fff?�  A���CY�                                    By��6  
�          @���=p�@�@*=qB
��C���=p�@P��?��
A���CG�                                    By���  
\          @����>�R?��R@0��Bz�CJ=�>�R@8��?�ffA�33C��                                    By��  
�          @�Q��0  @�@%�BG�C
���0  @N�R?��HA���Ck�                                    By�(  
�          @�G��Q�@G
=@ ��B��B����Q�@u�?���Ah  B��                                    By�%�  
�          @�
=��@AG�@
=B��B����@l��?��
AY�B���                                    By�4t  T          @��R��33��?޸RA�C5���33?��?�33A�=qC,O\                                    By�C  
�          @�Q���{<��
?�z�A���C3�R��{?&ff?��
A�ffC+�                                    By�Q�  T          @��
������
?�=qA�
=C5���?!G�?�(�A�33C+�                                    By�`f  "          @��\���
���?��RA�C6#����
?�R?��AÅC+u�                                    By�o  
Z          @��\���H��G�@�\A�(�C5�����H?.{?�
=AǮC*�
                                    By�}�  T          @�=q���=#�
?���A���C3s3���?G�?��A���C)0�                                    By��X  T          @�(��z=q<#�
?�33A��
C3�
�z=q?=p�?�  A�ffC)B�                                    By���  
(          @���~�R�#�
?�  A��C6G��~�R?�?�
=A�G�C,xR                                    By���  �          @�33�w�����?��HA�
=C5}q�w�?(��?���A��
C*G�                                    By��J  
�          @�(��z=q��=q?�(�A��
C8�z=q?�\?�A�ffC,��                                    By���  
�          @�G���Q콏\)@A�C4�3��Q�?=p�?���A�ffC)�                                    By�Ֆ  �          @�Q��qG�?�
=@	��A���C"���qG�?�?��
A�33C�q                                    By��<  "          @�G��j�H?�{@Q�A�=qCY��j�H@�
?���A�{Cٚ                                    By���  �          @�
=�i��?��@�A�  C��i��@{?�=qA�  C��                                    By��  �          @��\�E?�ff?�33A�G�C���E@��?�ffApz�C5�                                    By�.  �          @r�\���@C33?uApz�B����@L(���z�����B�W
                                    By��  
�          @z=q��ff@<(�?���A���B�p���ff@[�?�A	�B�                                    By�-z  
�          @}p���Q�@hQ�?uAb=qB�\)��Q�@mp������HB׊=                                    By�<   T          @��\�&ff@h��?�Q�A�(�B�.�&ff@�Q�>\)?��RB�\)                                    By�J�  �          @�33��Q�@aG�?�Q�A���B�LͿ�Q�@y��>8Q�@%�B��                                    By�Yl  T          @��\�J=q@l(�?�33A��B�G��J=q@|(��\)�G�B�                                    By�h  �          @\)���@mp�?��RA��B�Ǯ���@y��������B���                                    By�v�  �          @~{�.�R@Q�?\A��\C
���.�R@2�\>�@�ffCW
                                    By��^  �          @�G���33@\(�?��A��B�W
��33@j�H������B�z�                                    By��  !          @�녿xQ�@l��?�33A��B�p��xQ�@w
=�Ǯ��G�B�Q�                                    By���  
�          @�(��&ff@|(�?\(�AAG�Bƙ��&ff@}p��@  �)p�Bƀ                                     By��P  
�          @�{�aG�@|��?�ffAh��B�#׿aG�@�G�������B̏\                                    By���  �          @���=�G�@~�R?Tz�A<��B�Ǯ=�G�@\)�J=q�333B�Ǯ                                    By�Μ  �          @��ÿ�  @a�?�G�A�{Bә���  @o\)�aG��Mp�B��H                                    By��B  
�          @�(���p�@|��?��Ap(�B��R��p�@�������B�u�                                    By���  
�          @�ff>u@���?���At��B�B�>u@�z�\)���RB�p�                                    By���  �          @x��?:�H@e?��A��B�{?:�H@o\)��Q����B�                                    By�	4  
�          @{��@��@Q�?���A�G�C�@��@��>8Q�@,(�C��                                    By��  "          @����5@
=?�G�A��CY��5@0��>�@�Cٚ                                    By�&�  �          @����L(�?�Q�?�Q�A���C�f�L(�@ff?\)A�C�{                                    By�5&  "          @�  �\��?�
=?���A��C!33�\��?�33?^�RAL��Cu�                                    By�C�  "          @tz��fff>�(�?��A��
C-(��fff?Y��?aG�AV{C&Ǯ                                    By�Rr  �          @�  �j�H?��\?���A�\)C$���j�H?�33?333A!p�C{                                    By�a  "          @����U�?��
?�Q�A�{Cc��U�@z�?s33AZffC+�                                    By�o�  T          @���Vff?�33?��A�(�CQ��Vff@�?�Q�A��HC�                                     By�~d  
�          @{��I��?���?�p�A�  CxR�I��@Q�?xQ�Af{C�                                    By��
  T          @����XQ�?У�?�G�A���CE�XQ�@(�?xQ�AY�C\                                    By���  T          @�p��_\)?���?˅A�  Ch��_\)@ff?Tz�A8Q�C                                    By��V  
�          @���o\)?���?�Q�A�G�C )�o\)?�?J=qA,  C&f                                    By���  
�          @���e�?�\)?�{A�33C���e�@G�?(�A�Cz�                                    By�Ǣ  "          @����Y��?�p�?�G�A��\C��Y��@>�@��Cs3                                    By��H  �          @���R�\?���?��
A��C
=�R�\@�?.{A33CO\                                    By���  T          @�33�W
=?�{?У�A���Cz��W
=@�?\(�AC
=C�=                                    By��  �          @w��`  ?�  ?��
Ay�C B��`  ?Ǯ>�ff@�  C�                                    By�:  T          @n�R�:=q?�?n{An{C���:=q@Q�=�Q�?�33C�=                                    By��  
�          @����N�R?���?���A���C���N�R@{?
=q@��C��                                    By��  
Z          @�\)�i��?��H?��
A�(�CE�i��?�Q�?Tz�A4��C��                                    By�.,  
�          @���^�R?���?�A��HC���^�R@ ��?.{A��C                                    By�<�  �          @�G��j=q?�\)?��HA���C�R�j=q@��>�p�@�z�C�                                    By�Kx  T          @����U@�H?�{ApQ�C��U@*=q=L��?5Cu�                                    By�Z  T          @����G
=@-p�?���Au�C
���G
=@<(��#�
��C��                                    By�h�  "          @�\)�E�@,(�?�ffAf�HC
�)�E�@8�ý�G���Q�C�)                                    By�wj  
�          @|���Dz�@p�?��Az�HC=q�Dz�@��=�Q�?�{Cff                                    By��  �          @����:�H@(�?�p�A�C5��:�H@.�R>#�
@�
C	�                                    By���  
�          @�=q�B�\@Q�?�Q�A�\)C  �B�\@)��>��@z�C
�H                                    By��\  T          @����<��@p�?�
=A��C0��<��@   >L��@>{CǮ                                    By��  �          @����dz�?��
?��A��
C G��dz�?��H?8Q�A$��Ck�                                    By���  
Z          @����5�@!G�?���A��C
T{�5�@5>L��@9��C�                                    By��N  �          @s33�1G�@(�?�\)A���C���1G�@#33>\@���C	T{                                    By���  
�          @u�+�@�H?�ffA���C	ٚ�+�@/\)>k�@W
=CQ�                                    By��  
�          @vff� ��@E�?\)A\)B�B�� ��@C33�8Q��1G�B���                                    By��@  T          @vff���H@[�=u?fffB��
���H@J�H��������B�\                                    By�	�  �          @~�R��=q@g��#�
��Q�B�G���=q@S�
��������B��                                    By��  �          @�=q��p�@l��>��@�33Bߔ{��p�@c33��{��ffB�.                                    By�'2  !          @��ÿ��
@`��?�  A�(�B�=q���
@mp�����p��B��f                                    By�5�  
[          @�=q�ٙ�@U�?���A��HB�#׿ٙ�@l(�>�?��B�                                     By�D~  �          @������@(��@ffA��
C#����@P��?\(�A@Q�B��                                     By�S$  �          @����(�@+�?�p�A�RCJ=�(�@P  ?:�HA#�
B���                                    By�a�  �          @�=q���R@)��@ffB�B����R@W�?��AyG�B�\                                    By�pp  �          @������@(Q�@$z�BC\���@\��?�ffA�(�B�                                    By�  "          @��ÿ�\)@(��@3�
B#B����\)@b�\?�G�A�33B�Ǯ                                    By���  "          @�(��z�@  @FffB5z�C}q�z�@S�
?�
=Aҏ\B��                                    By��b  �          @�33�!�?�@;�B*��C�H�!�@;�?�z�A��
C��                                    By��  T          @��ÿz�H@N{@%�B  Bճ3�z�H@\)?���Aip�Bπ                                     By���  "          @��׿p��@p�@?\)BBB��p��@\��?�  AЏ\B�\)                                    By��T  
�          @�
=���
?��@`��Bdp�B��׿��
@HQ�@p�B�B�z�                                    By���  
�          @�G��"�\?�(�@N{BCz�Cff�"�\@=q@p�BffCxR                                    By��  "          @�ff�0  ?�(�@.{B$=qC��0  @(�?�33A�=qC
xR                                    By��F  	�          @�G��2�\?���@*=qBffC�H�2�\@.�R?ٙ�A�{C��                                    By��  T          @��H�<��?\(�@5B+  C#�q�<��?�
=@  BG�C�                                    By��  "          @����?\)?�ff@6ffB'�
C �3�?\)@
=@(�A���C�{                                    By� 8  
�          @��H�1G�?���@J=qB:CE�1G�@�@��B	(�Cu�                                    By�.�  �          @��\�(�?�(�@VffBKz�Cp��(�@{@$z�BQ�C��                                    By�=�  
�          @�  �0��?�(�@>�RB2�HC
�0��@z�@\)A��RC�H                                    By�L*  
�          @�\)�*�H?�(�@:�HB.�HC{�*�H@!�@z�A�\)C��                                    By�Z�  T          @�=q�(Q�?У�@?\)B0(�C��(Q�@,��@�A��HC0�                                    By�iv  !          @�=q�(�?��H@J=qB<�C
.�(�@Dz�@ffA�33B���                                    By�x  
�          @�(��У�@ff@;�B6B�uÿУ�@U?�p�A�  B�{                                    By���  T          @��`  ?fff@ffA�
=C%�)�`  ?�?�ffA��HC�                                     By��h  
�          @���Y��?���?��A��
CO\�Y��@   ?�
=A���C�\                                    By��  T          @~�R�9��?��?�{A�ffC��9��@�R?p��A]C��                                    By���  �          @o\)?��@2�\��33����Bd�H?��@����z�BE{                                    By��Z  �          @�G�@Z=q?aG�����G�AhQ�@Z=q<������R>�                                    By��   �          @���@X��?�{�����\)A�
=@X��?z��33��ffA�                                    By�ަ  
�          @�G�@P��?�G���ff����A�p�@P��?(��������A6=q                                    By��L  "          @�=q@(�?+��H���LffAw�@(��&ff�H���L��C��H                                    By���  �          @���@  >����Z�H�a��Aff@  ��=q�QG��R\)C�&f                                    By�
�  
Z          @��@z�?��^�R�i�
Ab�H@z�n{�Y���`�
C���                                    By�>  �          @���@�
?Q��Mp��R�A�G�@�
����QG��X�C�W
                                    By�'�  T          @�  @
=?����C33�F��A�
=@
=����N{�Vp�C��                                     By�6�  �          @���@�?���@���B=qA�33@��L���Q��ZQ�C�o\                                    By�E0  �          @�z�@,(�?��6ff�0�A��@,(���Q��E��C\)C�{                                    By�S�  
�          @��H@�
?����<(��9p�B
�
@�
>�=q�U�\�@Ӆ                                    By�b|  
�          @�  @ff?�
=�2�\�1B*\)@ff?!G��U��b�HA���                                    By�q"  T          @���@�
?����&ff�"�B z�@�
?:�H�J=q�R{A��                                    By��  "          @���@Q�@��*=q�%�
B3z�@Q�?\(��R�\�\=qA�                                      By��n  "          @��
@
=?�
=�;��6\)B)��@
=?z��]p��fz�Au                                    By��  T          @�z�@�\?�(��W��[�A�z�@�\��  �e��p=qC�t{                                    By���  �          @�G�?�z�@9���	���
=Bp�H?�z�?��
�H���Q
=B;�H                                    By��`  �          @��?�G�@<(��'
=� G�B�\?�G�?�\)�e��w�\Bhz�                                    By��  "          @}p�?�(�@dz�0���!p�B�G�?�(�@>�R�����B(�                                    By�׬  !          @\)?��@c33��Q�����B�� ?��@*=q�0  �.ffB�                                    By��R  U          @}p�?��@<�Ϳ�=q��ffB��=?��?����7��H�\BZ=q                                    By���  �          @~{?�  @h�ÿaG��Q��B�33?�  @>{���33B��                                    By��  
�          @~{>�\)@r�\?n{A\��B�p�>�\)@vff�0���!p�B��=                                    By�D  
�          @~�R>��
@vff?Tz�A@��B�k�>��
@w
=�O\)�=G�B�p�                                    By� �  S          @�  ����@g
=?У�A��B��\����@}p�<��
>���B���                                    By�/�  U          @\)��{@a�=�\)?��B���{@P�׿�\)��{Bم                                    By�>6  �          @�=q�B�\@`��?�A�33B�
=�B�\@|��>��@mp�B�\)                                    By�L�  T          @��\���@)��@-p�B(B�W
���@aG�?�33A�ffB�
=                                    By�[�  
�          @�p����@:�H@-p�B$ffB�Ǯ���@p��?��A�B�Q�                                    By�j(  T          @�33��G�@`��?�Q�A�B����G�@�  >�33@��B�8R                                    By�x�  "          @�z�Ǯ@`  @G�A�
=B��)�Ǯ@���>�(�@���B�33                                    By��t  T          @�p����
@l��?�{A�
=B�𤾣�
@�z�>L��@.�RB��f                                    By��  �          @��\<�@`��@ ��A�{B���<�@���>��@���B��                                    By���  T          @�녿���@Dz�?�\A��
B�{����@a�>�p�@��B�W
                                    By��f  
�          @���� ��@)��?���A�Ch�� ��@Dz�>Ǯ@��\C5�                                    By��  �          @|���G�@;�?�ffA�z�B����G�@L��    �L��B��3                                    By�в  �          @w�>�@n{�z��B�(�>�@J�H��\�Q�B�z�                                    By��X  
Z          @mp�?J=q@HQ�Ǯ��{B��q?J=q@{�,(��C{B�                                    By���  �          @]p���ff?�녾.{�˅C���ff?xQ��R���C	E                                    By���  �          @P  �A�?333���G�C'
�A�?
=������G�C){                                    By�J  �          @L�Ϳ�\)@33�Y���33C���\)?�G����H�C��                                    By��  �          @_\)��{@@�׿
=q��RB�=q��{@!녿��H���
B��
                                    By�(�  �          @a���
?�=q@��B �CaH��
@?�{A���CL�                                    By�7<  �          @]p���(�@#33?�p�A�=qB���(�@5�=�Q�?��RB�=                                    By�E�  �          @(�����?�ff?=p�B�B�����?�G�>�  A.=qB�8R                                    By�T�  �          @L(�@
=?�p���
=��Q�B�@
=?E���z��!�\A�33                                    By�c.  �          @Y��@{?�\��6��AMG�@{���H��7=qC��3                                    By�q�  T          @i��?޸R>8Q��E�r
=@�33?޸R����8���Z��C��                                    By��z  
�          @��?�?�(��e��jffB��?���{�qG����C��                                    By��   �          @��\?޸R�8Q��g��\C�f?޸R����O\)�Vp�C�W
                                    By���  �          @}p�?�녿�
=�Q��iQ�C�&f?���=q�!G��#(�C�"�                                    By��l  �          @��?�
=����hQ��n�\C��?�
=�*�H�2�\�%�C��                                    By��  �          @��?�\)����e�]p�C�S3?�\)�K��!G���HC�y�                                    By�ɸ  �          @�G�?�  �����^{�W�\C���?�  �Dz����	��C��{                                    By��^  �          @��?�������X���Jp�C�  ?����g��ff���C��                                    By��  �          @�  ?!G��Mp��Dz��,�C��{?!G�����  ��(�C�O\                                    By���  �          @��?5�b�\�)���\)C��?5��녿}p��L(�C��f                                    By�P  �          @��?+��g�����C�@ ?+���G��@  ���C�p�                                    By��  T          @|(��Q��
=?�p�B
(�C^@ �Q�u@&ffB?�CL\)                                    By�!�  �          @�Q�Q��L�Ϳ�33��  C���Q��`  <��
>�=qC�j=                                    By�0B  �          @x�ÿ���e��&ff���C}#׿���b�\?\(�AQC|�                                    By�>�  T          @�G�@
=?�ff�S33�N33Bz�@
==��
�h���o�R@�R                                    By�M�  �          @�z�@
�H?c�
�n{�g=qA�ff@
�H�333�p���kG�C�R                                    By�\4  �          @�z�@{?=p��K��L
=A��
@{�(��Mp��Nz�C��                                    By�j�  �          @|��@�R?�{�2�\�733A��@�R��G��@  �I�
C���                                    By�y�  �          @z�H@   ��33�.{�@G�C��f@   �Q����\)C��                                    By��&  �          @r�\?���+�����C��3?���Y����ff����C�`                                     By���  �          @j�H?��
��H��
��RC�!H?��
�H�ÿ������C��                                    By��r  �          @c�
?�p��������C��?�p��A녿}p���33C��
                                    By��  �          @AG�?�{�p���ff� �\C�^�?�{�)�����&=qC�e                                    By�¾  �          @a�?�녿�(�����,G�C�޸?���"�\���R�ɅC�XR                                    By��d  �          @J�H?�G����Q����C�"�?�G��4z����/�
C��                                    By��
  �          @�ff@,���\)�7����C���@,���\(���{��  C�{                                    By��  T          @��H@Dz�����7���C��R@Dz��W
=��33���C�5�                                    By��V  T          @�33@E����.{�C��q@E��@  ������C��                                     By��  �          @���@@  ����4z��Q�C��R@@  �7
=������  C�/\                                    By��  �          @��R@I����z��+����C��@I���&ff���
��=qC�:�                                    By�)H  �          @���@g
=��G��p��
=C�c�@g
=��ÿ޸R��=qC���                                    By�7�  �          @�33@g��Ǯ�4z��z�C�T{@g��$z������  C�P�                                    By�F�  �          @���@w
=���R�/\)��C�
@w
=�\)� ����p�C��\                                    By�U:  �          @��@�=q�s33�{���HC�w
@�=q��{�����Q�C��{                                    By�c�  �          @��@�G��Tz��!�����C�0�@�G����
��p�����C�
                                    By�r�  �          @�{@��\�&ff�{���
C�y�@��\���R��\��33C���                                    By��,  �          @��@���k��   ���HC���@�����\�޸R��  C���                                    By���  �          @��
@�{�L���   ���
C��q@�{�z�H��  ��z�C���                                    By��x  T          @�{@�{>�
=�����]G�@�{@�{�u��
=�m�C��                                    By��  �          @�{@�?zῃ�
�PQ�@�ff@�=�G���
=�nff?�(�                                    By���  �          @�
=@��?#�
�^�R�+
=A Q�@��>u��ff�P  @?\)                                    By��j  �          @���@�?333�333�=qAQ�@�>�Q�k��1��@��                                    By��  �          @�=q@���>\)�0����?�Q�@��׾���0���\)C��                                    By��  �          @��@��>aG��E��z�@(Q�@����G��L�����C�Q�                                    By��\  �          @��\@�Q�>Ǯ�(����@�ff@�Q�=�G��8Q����?�{                                    By�  �          @�=q@�ff>��
�Tz���
@vff@�ff�#�
�c�
�+
=C��)                                    By��  �          @��@�G�?8Q쿅��K\)A(�@�G�>u���R�s�
@C�
                                    By�"N  �          @�=q@���?�����\)�S�
Ag�@���?}p��=p����A@                                      By�0�  �          @�G�@���?��
���Ϳ�  A�p�@���?�{�5���A�                                    By�?�  �          @�  @��?�(���\)�W
=A��H@��?�ff�B�\��HA��                                    By�N@  T          @��@��?��
=���?�p�A�33@��?���R����A�33                                    By�\�  �          @�(�@��\?�33    �#�
A��@��\?�G��(����A�(�                                    By�k�  �          @��@�ff?��>��?�=qA�@�ff?����R��  A���                                    By�z2  �          @�z�@�ff?�(�=#�
?   A�\)@�ff?�=q�@  �p�A��                                    By���  �          @��@�  ?��;�p���(�A�z�@�  ?�ff��=q�X  A�(�                                    By��~  �          @��@�p�?�=q>\)?�
=A���@�p�?�p������\)A�
=                                    By��$  �          @�=q@�{?�G��Ǯ����A�33@�{?��H����L��A���                                    By���  �          @�=q@�(�?�
=���H��=qA�  @�(�?��ÿ�(��jffA��                                    By��p  �          @�=q@�z�?��
=q�θRA��R@�z�?����G��s
=A��R                                    By��  �          @��@�33?�G���z��Y��A��H@�33?��\�\(��$Q�Au�                                    By��  �          @���@�G�?��H�����A�Q�@�G�?��u�8��Ag�                                    By��b  �          @�  @�p�?�
=>\@�G�A��\@�p�?�Q쾨���|(�A��                                    By��  �          @�{@��R?�p�>�z�@a�A�  @��R?�
=����  AŅ                                    By��  T          @�  @�33?�논#�
��Q�A�(�@�33?�p��B�\��A��
                                    By�T  T          @�\)@�z�@G�>W
=@$z�A�=q@�z�@
�H�5�
ffA�\)                                    By�)�  �          @��@z=q@{����(�B�@z=q@�����=qA�                                    By�8�  �          @�p�@Vff@ff�.{���B  @Vff@�����w33A�
=                                    By�GF  �          @u@L��@ �׾�(��У�B @L��?�
=��������A�G�                                    By�U�  �          @1�@�
?��5�u�A�G�@�
?J=q��\)�ď\A�                                    By�d�  �          @
=q>�z�������.C���>�z῕���H�3�RC��                                    By�s8  �          @7
=�\(�������\�RCn&f�\(���Ϳ�����\Cx�f                                    By���  �          @>{�u��  �(Q�Q�C���u�(����$�C��                                    By���  �          @C33�(���G��,(��z33Ct#׿(��{��(��"G�C~�f                                    By��*  �          @*�H=L�Ϳ�33���� C�#�=L��� �׿޸R�#�RC���                                    By���  �          @0��?�Ϳ��H�Q��r��C�:�?���33�ٙ��C���                                    By��v  �          @'�>\��  �  �o�C�xR>\�녿������C�J=                                    By��  T          @$z�>�Q쿌����\�|�
C�)>�Q��z���"ffC�Y�                                    By���  �          @$z�>��!G��(��)C��>���ff���R�K�\C��=                                    By��h  �          @/\)?�
=<#�
�(�G�>�{?�
=�u�\)�_z�C���                                    By��  �          @<��?�(�?����
=q�0z�B2�H?�(�?�=q���\��
=B�H                                    By��  �          @�{@*�H@?\)?�33A�B@�@*�H@L�ͽ����HBH��                                    By�Z  �          @��@<(�@=p�?=p�A (�B4��@<(�@@�׾��H���
B6�\                                    By�#   �          @��@B�\@A�>\@�=qB3�@B�\@;��W
=�4Q�B/�H                                    By�1�  �          @���@@��@?\)?.{A�B3�@@��@@�׿�����HB4(�                                    By�@L  �          @���@N{@+�?333A
=B  @N{@/\)������\B!p�                                    By�N�  �          @��@B�\@9��?h��AC\)B.�@B�\@A녾�������B3z�                                    By�]�  �          @�G�@QG�@&ff?}p�AU�B�@QG�@1녽�G����RB!��                                    By�l>  �          @��@k�@G�?W
=A/33A�G�@k�@=q������B�                                    By�z�  T          @�Q�@e@{?�=qA�z�A��@e@*=q?
=@�
=BG�                                    By���  
�          @�ff@[�@��?�A��A���@[�@,(�?Y��A0z�B=q                                    By��0  �          @�
=@^�R@�?�z�A�
=A�Q�@^�R@'�?xQ�AG�
B�
                                    By���  �          @��@`��?��R?�(�A��A�z�@`��@&ff?��
AT��B33                                    By��|  �          @��R@S�
@
=q@�\AڸRB�@S�
@2�\?��AW�B ff                                    By��"  �          @�{@HQ�?��H@(�A��A�R@HQ�@�?���A�{B��                                    By���  �          @�  @Fff@ ��@A��B�@Fff@*�H?��Ay�B"�H                                    By��n  �          @�@8��@�?��A�\)B��@8��@8Q�?Tz�A7\)B3�R                                    By��  �          @��@
�H@J=q?�G�A���B]�H@
�H@`  >#�
@�Bh�                                    By���  �          @�33?p��@�{>�(�@��B���?p��@��׿��R��p�B���                                    By�`  �          @��
?�{@�G�?h��AA�B��q?�{@�=q�:�H�ffB�\                                    By�  �          @�33?���@~{?��AiG�B�W
?���@��\�
=q���
B�L�                                    By�*�  �          @���?��
@~{?z�HAPQ�B�Ǯ?��
@�G��!G����B�u�                                    By�9R  T          @�(�?}p�@y��?�p�A��B��{?}p�@�p�������RB�W
                                    By�G�  �          @��
?@  @�p�?�ffA^�HB���?@  @�  �&ff�Q�B�                                      By�V�  �          @��\>�
=@�
=?Q�A.�\B���>�
=@��R�aG��<(�B��{                                    By�eD  �          @���?k�@��?.{A��B�k�?k�@�p����\�VffB�
=                                    By�s�  �          @�p�?u@�G�>W
=@333B���?u@�G����H����B��                                    By���  �          @�\)?��
@�녾�Q���(�B�33?��
@u����R��\)B�
=                                    By��6  �          @��?�p�@��Ϳ#�
�B��)?�p�@e��(���  B���                                    By���  �          @�=q?(��@�(��:�H�B��R?(��@a����� ��B��3                                    By���  �          @���?���@xQ�(�����B�\)?���@U��
=��  B�ff                                    By��(  �          @�?�  @��R��
=��\)B��f?�  @n{� ���ظRB��)                                    By���  �          @�Q�?�  @�G�������B��\?�  @b�\�   ��=qB�{                                    By��t  �          @{�?fff@n{�B�\�4��B��?fff@H���Q���B�(�                                    By��  T          @W�?0��@L(��Y���j�HB��R?0��@'�� �����B���                                    By���  �          @�ff?aG�@y����  ���HB��=?aG�@G��)����B�z�                                    By�f  �          @��
?L��@u������(�B�aH?L��@E�%���HB��)                                    By�  �          @���>��@}p���
=��ffB�(�>��@G
=�5�(�\B��                                    By�#�  �          @�\)>Ǯ@�p�>�G�@���B��H>Ǯ@�  ���
���B�z�                                    By�2X  �          @��>��@�
=?#�
A�RB�k�>��@�(���{�d  B�G�                                    By�@�  �          @�=q>��@�\)?L��A"ffB�Q�>��@�ff�xQ��C�
B�B�                                    By�O�  �          @��=#�
@�녾�{��=qB���=#�
@g���{��B��)                                    By�^J  �          @�G�?&ff@s�
����v{B���?&ff@G�����B�z�                                    By�l�  �          @��?�\@�z�>#�
@��B�  ?�\@x�ÿ�
=��Q�B��                                    By�{�  �          @�
=��=q@U?�\A�Q�B����=q@p��>�p�@���B��                                    By��<  
�          @�ff���
@dz�@&ffB�B�B����
@�G�?�ffA[
=B��=                                    By���  �          @��\>k�@W�@)��B�B�
=>k�@��
?�A|��B�z�                                    By���  T          @�  >u@�(�>W
=@:�HB�k�>u@z=q��\)��ffB�
=                                    By��.  �          @�ff>Ǯ@�(��\���B�(�>Ǯ@P  �>{�)  B�8R                                    By���  �          @�Q�>L��@mp��˅���B��f>L��@5��8Q��5z�B�                                      By��z  �          @|(�>�  @Vff����33B�� >�  @=q�=p��J��B�33                                    By��   �          @�{?(�@y�����R��Q�B��?(�@7
=�S�
�B\)B���                                    By���  �          @�>#�
@mp�������B�#�>#�
@!��g��\  B��H                                    By��l  �          @�\)>#�
@l���   �{B�\>#�
@\)�mp��`\)B���                                    By�  �          @����
=@g
=��R�	\)B����
=@=q�j=q�`�
BǊ=                                    By��  �          @�
=��z�@n{���p�B��H��z�@"�\�i���\G�B�                                      By�+^  �          @�{=���@;��H���<�B�.=���?��
��Q�\B��R                                    By�:  �          @���#�
@7��Vff�Ep�B�8R�#�
?�z���{��B���                                    By�H�  �          @����
@,(��`���RG�B��ü��
?�
=��Q�{B�.                                    By�WP  �          @�G���@33�g
=�e�B����?J=q���Ru�BŨ�                                    By�e�  
�          @��H?(�@�\�qG��q�HB��=?(�>��H��Q�{BQ�                                    By�t�  �          @�G�>��@�R�W��UG�B�.>��?�������B�                                      By��B  �          @����.�R@#�
�Ǯ���RC�\�.�R@�R�������
C�R                                    By���  �          @x���L��@������RC���L��?�\���\���C                                    By���  �          @x�ÿУ�@?\)��p���\)B�\�У�@���.{�5\)B���                                    By��4  �          @�����G�?p���n{�fC#׿�G���\�s33�CF��                                    By���  �          @�33��  �.{����C=�׿�  ���H�tz��z33Co�f                                    By�̀  �          @���L��>.{��33�C'޸�L�Ϳ�{�w��qCo��                                    By��&  �          @�z�?Tz�?L���s33aHB/�?Tz�(���u�  C���                                    By���  �          @�����\<��{�8RC2s3���\���j�H�z=qCd:�                                    By��r  �          @fff�Tz�?����<(��q��B�\�Tz�>���QG�=qC"��                                    By�  �          @p  �G�@A녿�{��{B�\�G�@{�'
=�?�HB��f                                    By��  �          @�������@`���	����ffBמ�����@{�R�\�I��B�
=                                    By�$d  T          @�{�.{@�33��Q����B��f�.{@Q��6ff�!\)B�z�                                    By�3
  �          @�zῚ�H@���=���?�ffB��
���H@�(��\��\)BԽq                                    By�A�  �          @��   @�=q?fffA3
=B�L��   @���(����RB���                                    By�PV  �          @��Ϳ��@z�@k�BS=qB�8R���@`��@$z�Bp�B�                                    By�^�  �          @�녿�z�@�\@]p�BR��C#׿�z�@J=q@{B
��B�ff                                    By�m�  �          @�Q���@/\)@O\)B7\)B��Ϳ��@n�R@ ��A�{B��                                    By�|H  �          @�p��k�@\)@>{B@�RBܙ��k�@Y��?�{A�=qB�Q�                                    By���  �          @��@l(�?�G��HQ��A��
@l(�?   �c33�.�@�ff                                    By���  �          @�ff@mp�@G��AG���A�=q@mp�?G��c33�,�\A=                                    By��:  �          @�
=@i��?�=q�S33���A�\)@i��>�\)�i���3�@��\                                    By���  �          @�{@qG�?��
�Z=q�$p�Au@qG���=q�c33�,��C���                                    By�ņ  �          @�@dz�?��
�A���HA�G�@dz�>����XQ��-33@�
=                                    By��,  T          @�p�@Z�H?����P���&z�A���@Z�H=L���`���7�?fff                                    By���  �          @�{@^{?���<����A�p�@^{?0���[��0ffA333                                    By��x  �          @��@S�
@(��#33���\Bff@S�
?����P���)Q�A��
                                    By�   �          @�p�@<��?޸R�aG��7G�A�=q@<��>�33�z=q�S33@�33                                    By��  �          @�ff@J=q?����3�
��A��@J=q?O\)�Tz��5�
Ag
=                                    By�j  �          @�33@P  ?�33�AG���HA�  @P  ?0���`  �9�
A>�H                                    By�,  �          @�p�@C�
?���Vff�+��A��H@C�
?\)�s33�J�HA%G�                                    By�:�  �          @��H@\��@33�)���A��@\��?xQ��Mp��'ffAz�\                                    By�I\  �          @�(�@Y��?���:=q�(�A�\@Y��?8Q��X���133A@��                                    By�X  �          @�  @K�@
=q�3�
�z�B�R@K�?��\�X���5��A��
                                    By�f�  �          @�{@[�?����+��G�A��@[�>����?\)�#�H@��\                                    By�uN  �          @��@z�?��|���k��AJ{@z�k��w��d�RC�9�                                    By���  �          @��
@Q�?(��y���g=qAg�
@Q�O\)�w
=�c�HC��3                                    By���  �          @��H@*=q?}p��e��N��A�\)@*=q��33�l���Xp�C�H�                                    By��@  �          @��@Fff@U��Q�����B<��@Fff@?\)�\��p�B0�                                    By���  �          @��\@@  @E��\)��p�B7(�@@  @���Q�����B��                                    By���  �          @�@7
=@5����  B3�@7
=?��K��,\)B��                                    By��2  �          @��@@��@\(���
=�f�HBC�\@@��@3�
�z�����B,33                                    By���  �          @��
@ff@'
=�0  �p�BL�@ff?�p��_\)�VB��                                    By��~  �          @��\@�@>{�
�H��p�BY�H@�@G��Dz��9p�B.ff                                    By��$  �          @l��?��@,(��	�����B��\?��?�G��<���U�BV�H                                    By��  �          @��
@Q�@6ff�  �=qBU
=@Q�?���Fff�>33B%��                                    By�p  �          @�=q@HQ�@5��ٙ�����B(�R@HQ�@�
�%�
p�Bz�                                    By�%  �          @��@Dz�@=p���
=����B/�
@Dz�@G��Q���G�B{                                    By�3�  T          @��@H��@7���33����B)�H@H��@p���
���RB\)                                    By�Bb  �          @��R@B�\@=p���\)��\)B1  @B�\@33�z���z�B\)                                    By�Q  �          @���@>�R@H�ÿ�����RB9�
@>�R@   �33���HB��                                    By�_�  �          @�\)@AG�@C�
������
B5z�@AG�@�H�G���p�B�H                                    By�nT  �          @�{@3�
@E��Q����\B>�
@3�
@������B!�H                                    By�|�  �          @�p�@Q�@C33��=q��G�B\(�@Q�@�R�1G��'�B9�                                    By���  �          @��@{@B�\�(���G�BKp�@{@ff�Fff�.�HB!(�                                    By��F  �          @���@<(�@?\)�ٙ���(�B5�H@<(�@{�'���RBG�                                    By���  �          @�Q�@W�@1G�����`��B  @W�@�R��p���
=B
=                                    By���  �          @�
=@hQ�@1녿�z��d  B�
@hQ�@p��33�͙�A��H                                    By��8  �          @��R@s�
@*=q�5�
=B��@s�
@  ��=q��
=A���                                    By���  �          @���@tz�@(�ÿ��
�G�
B
�\@tz�@Q��\)��G�A���                                    By��  �          @�G�@tz�@*=q�����T��B�@tz�@Q�������A���                                    By��*  T          @���@\(�@=p��fff�2�\B"��@\(�@{��������B�
                                    By� �  �          @���@J=q@H�ÿ\(��.ffB3
=@J=q@*=q�����G�B �                                    By�v  �          @��@C�
@6ff���H���B+��@C�
@G��
=�癚B(�                                    By�  �          @�  @Dz�@8Q�����=qB,�@Dz�@#33����(�B�R                                    By�,�  �          @��
@>�R@0  �0���p�B*@>�R@ff��=q��=qB33                                    By�;h  �          @w�@z�@*=q��ff��p�BC��@z�?�p��
=��
B"G�                                    By�J  �          @���@,(�@5�������\B:
=@,(�@�R�
�H��33B�R                                    By�X�  �          @��\@�
@E��33���Ba=q@�
@
=�&ff��BC33                                    By�gZ  T          @\)@  @5��33��z�BN=q@  @
=� ����B,��                                    By�v   �          @��
@!�@C33����33BIz�@!�@���0����B'{                                    By���  �          @�ff@+�@0  �˅��p�B7(�@+�@�
��H�=qB\)                                    By��L  �          @���@:=q@&ff��(���\)B'�@:=q?��H�  �{B�R                                    By���  �          @��R@L��@�������B(�@L��?�\�ff��RA��                                    By���  �          @��H@Y��@z΅{��{B	G�@Y��?޸R��
��=qA�ff                                    By��>  T          @��
@H��@.�R���{
=B$  @H��@�� ����BQ�                                    By���  �          @�(�@AG�@=p������i��B1�@AG�@��G���z�B{                                    By�܊  �          @��
@Q�@   ������  BQ�@Q�?�����
=A���                                    By��0  �          @��
@W�@=q�������B  @W�?�=q��
��A�                                    By���  �          @��@G
=@.�R��ff����B%�@G
=@	������\B
��                                    By�|  �          @�33@G
=@6ff��G��V�HB)��@G
=@
=����Σ�B��                                    By�"  �          @��@=p�@J�H�p���C
=B<(�@=p�@,(���
=��(�B)33                                    By�%�  �          @���@�@N�R������ffBV�H@�@&ff�����B>ff                                    By�4n  �          @�@   @G
=��33�˙�BL�H@   @�
�3�
�(�B+Q�                                    By�C  �          @�G�@<��@Vff�G���BB��@<��@:=q��=q���\B2��                                    By�Q�  �          @���@8Q�@Fff����f�HB<33@8Q�@$z���\�ޣ�B&��                                    By�``  �          @�  @33@33�,���$��BAQ�@33?�ff�S33�V��Bz�                                    By�o  �          @^{?��
?����2�\�`�Bb�H?��
?z��I����A�z�                                    By�}�  �          @��@
�H?��H�G��;G�B(�@
�H?W
=�e��d  A�(�                                    By��R  �          @�z�@
=@
=q�J=q�9  B6�\@
=?��\�l(��f{A�z�                                    By���  �          @�ff?�(�?���u�y�A�z�?�(���  �~�R�)C�Ǯ                                    By���  �          @�\)?��?��H�N{�a33BR�?��?z��fffǮA�
=                                    By��D  �          @��\?�p�@@  �����G�Bbp�?�p�@\)�0  �*��BA��                                    By���  �          @��@%@33�G��(�B&��@%?�
=�l���Q�RAÙ�                                    By�Ր  �          @�{@Y��?����)���

=A�@Y��?aG��E�%�Ag�                                    By��6  �          @��R@O\)?�Q��I���)z�A��@O\)=��W
=�8=q@	��                                    By���  �          @��R@A�?xQ��\���={A�{@A녾8Q��e��F��C�N                                    By��  �          @�@/\)?B�\�mp��RffAx��@/\)���p���VffC�5�                                    By�(  �          @��@ ��?�G��p���Y33A���@ �׾u�x���d�C�>�                                    By��  �          @�G�@\)?+�����r�RA��R@\)�0������rffC�xR                                    By�-t  �          @�G�@�=�������G�@2�\@���G�����mffC�\)                                    By�<  �          @�33?�33?�
=��
=�r=qB33?�33<����R��?\(�                                    By�J�  �          @�\)@G�?�{��Q��k\)B
=@G�>8Q�����@��\                                    By�Yf  �          @�{?�
=?��������qz�B+�R?�
=>aG����\.@���                                    By�h  �          @��H?�{?���`  �l�\B�?�{>.{�p  W
@ƸR                                    By�v�  �          @���?�p�?���R�\�S��B7�\?�p�?+��l(��}
=A�Q�                                    By��X  �          @q�@�?�  �,���6Q�A�  @�>�33�=p��M��A
=                                    By���  �          @n{?�?���:�H�R
=BQ�?�>����K��o�\A'\)                                    By���  �          @���@�?����c�
�X��B�\@�>\�w��w�A*ff                                    By��J  �          @��?�=q?�ff�mp��kz�BVQ�?�=q?\)���H{A�G�                                    By���  �          @��H?��
?��\���g
=Bw  ?��
?@  �w��=B��                                    By�Ζ  �          @���?   @�\�c�
�m�B�33?   ?W
=������Bk�R                                    By��<  T          @����Q�@��n{�g�B�����Q�?�����  �3B�(�                                    By���  �          @��>�  @33�n{�h\)B���>�  ?�ff����{B�33                                    By���  �          @�
=?��R?У��W��j��BS=q?��R?��mp�33A�                                    By�	.  �          @�33@\)��  �\(��B�RC�,�@\)��\�=p��'\)C�b�                                    By��  �          @���@w��aG����
�n�\C�e@w����k��S\)C�(�                                    By�&z  �          @���@tz�u��ff����C��{@tz��ff��(���33C���                                    By�5   �          @|(�@r�\>k��xQ��dz�@Z�H@r�\���
�}p��j{C�g�                                    By�C�  �          @���@r�\>�����R��(�@{�@r�\����G�����C�
=                                    By�Rl  �          @~�R@dz�>�
=��
=���H@���@dz��G���p����HC�
                                    By�a  �          @~{@W�>\�G����@�Q�@W��u��\����C��                                    By�o�  �          @~�R@XQ�>#�
�33���
@/\)@XQ��� ����C�
                                    By�~^  �          @w�@Vff>.{����Q�@:�H@Vff��녿����33C�y�                                    By��  �          @�p�@e�\)�����ffC���@e���Ϳ�����RC�w
                                    By���  �          @�Q�@��ÿ���{��=qC���@��ÿ��Ϳ�Q��uC�7
                                    By��P  �          @��@�  �(�ÿ޸R��=qC��@�  ��33���H��
=C�g�                                    By���  �          @�@�{���������C�P�@�{�5�����ř�C�'�                                    By�ǜ  �          @��\@xQ�>��Ϳ�z��Џ\@�{@xQ�8Q��Q��ԏ\C���                                    By��B  "          @��@mp�>��R�\)���
@�G�@mp���p���R����C�/\                                    By���  �          @���@n�R>�p������@��R@n�R��p������RC�.                                    By��  T          @��@]p�?
=�!G����A�R@]p��8Q��%�  C�}q                                    By�4  "          @o\)@;�?
=q�  ��A&�\@;�����33��C���                                    By��  �          @{�@N{?�\�
=q�
=A�\@N{����p��	��C���                                    By��  T          @~{@W�>��G���
=A ��@W����z���  C��)                                    By�.&  
�          @qG�@C�
?k��   � (�A�  @C�
>������p�@���                                    By�<�  
�          @u@QG�?\(���ff���
AmG�@QG�>�����(����
@��                                    By�Kr  
�          @�ff@W
=?�\)�33���A�@W
=?8Q��
=�
{A@��                                    By�Z  �          @�33@=p�?��R�33�  A�p�@=p�?B�\�(Q��"�Ae��                                    By�h�  �          @e�@J�H?(���G��ȣ�A-G�@J�H>\)��\)�؏\@%                                    By�wd  �          @��@��\    �����(�C���@��\��R�33����C�#�                                    By��
  �          @�(�@�=�G����H��\)?��
@�����z����
C��q                                    By���  �          @��@u=��
��=q��
=?�(�@u�\������C�4{                                    By��V  �          @mp�@`  >B�\�����(�@G�@`  �\)��33��
=C��3                                    By���  T          @Mp�@>�R>��ÿ�{��  @�(�@>�R    ��33���=�\)                                    By���  �          @`  @N{?�R��Q���
=A/�@N{>�������33@�(�                                    By��H  �          @b�\@J�H?����(���  A+�
@J�H>#�
��=q���
@7�                                    By���  �          @aG�@I��?�Ϳ\���A�@I��=�Q��{��z�?�Q�                                    By��  T          @a�@HQ�>�׿�ff��=qA	G�@HQ�<��
��\)��z�>�z�                                    By��:  "          @C33@.�R>���������@�33@.�R���
�����\)C�/\                                    By�	�  T          @6ff@\)>��R��=q��{@��@\)���
��\)��Q�C�!H                                    By��  �          @Vff@7�?������\A$��@7�=#�
��(���\)?Tz�                                    By�',  �          @9��@'�>����
=��=q@U�@'��8Q쿗
=��p�C��)                                    By�5�  "          @E�@\)?=p���33��A���@\)>�  ����p�@��                                    By�Dx  �          @`��@%?�R�33��\AX(�@%<�������?�                                    By�S  T          @w
=?@  @8Q����{B��f?@  @
�H�5�K�HB��                                    By�a�  �          @�z�>�(�@q���\��B���>�(�@C�
�@  �0�B���                                    By�pj  �          @�
=?�p�@^�R�{���Bqz�?�p�@*=q�U�4�RBUp�                                    By�  �          @��\?��@`  �%��
�B�  ?��@*=q�\���E\)B
=                                    By���  �          @�=q?�z�@'��H���>p�B�ff?�z�?�z��o\)�vQ�B\z�                                    By��\  �          @�ff?�z�@&ff�a��C=qBe�\?�z�?��
��33�tz�B*��                                    By��  
Z          @�33?��H@U�Y���(B{=q?��H@����^BTQ�                                    By���  
Z          @���?�z�@H���\(��,=qBj��?�z�@����_33B>Q�                                    By��N  T          @��?��@l(��E���B{�\?��@-p��~�R�I
=B\�\                                    By���  
�          @�  ?�ff@X���S33�&(�B��3?�ff@����H�]ffBb��                                    By��  �          @��
?��@\���?\)���B�8R?��@!G��s�
�N�HBc                                    By��@  T          @�33?�
=@hQ��:�H��RB�\?�
=@-p��r�\�PQ�B�                                    By��  T          @�=q>�  @�p���=q����B���>�  @u�-p���B���                                    By��  T          @��\���\@���?O\)A$(�B�\���\@��\��p�����Bԣ�                                    By� 2  
�          @�
=�aG�@�p�?��A�(�Bʏ\�aG�@��
=�G�?���Bɞ�                                    By�.�  
Z          @����\@`��@�A���B�W
��\@z�H?��\AP��B��                                    By�=~  �          @�z��Q�@c33@ffA�33B�LͿ�Q�@~�R?�=qA[
=B���                                    By�L$  
�          @�녿�(�@g�?�
=A���B�  ��(�@�  ?fffA7\)B�Q�                                    By�Z�  "          @��H��(�@��?��\A��B�LͿ�(�@���>\)?޸RBٮ                                    By�ip  �          @����G�@�  ?(�@�RB�ff��G�@�  �
=��  B�aH                                    By�x  T          @�33���@��\?#�
@�(�B����@��H�z���ffBճ3                                    By���  
�          @��
��z�@�\)?�Q�Ad  B��f��z�@�zἣ�
���B�Ǯ                                    By��b  
�          @�33��z�@�?��A�{B����z�@�{>Ǯ@��B�{                                    By��  
�          @�  ���H@���?��An�\B�����H@�p��#�
��Q�B��                                    By���  T          @�����G�@�z�?���A��B��f��G�@��H>L��@ ��B�k�                                    By��T  T          @�33�L��@��>�G�@��B�(��L��@�ff�=p��Q�B�W
                                    By���  T          @�(���\)@�33>�  @FffB���\)@�Q�s33�=G�B�(�                                    By�ޠ  	�          @�
=�(�@�33?�p�A�Q�B��H�(�@���?\)@�Q�B��H                                    By��F  	�          @���.{@���?�Q�A��B��f�.{@��H?�@�G�B���                                    By���  
�          @��H���
@.{@z=qBI{B�8R���
@hQ�@E�B\)B��                                    By�
�  
�          @�z��G�@;�@`��B6��B��ÿ�G�@n�R@(��B�B�=                                    By�8  T          @��Ϳ���@L(�@C33B%=qB�𤿬��@w
=@��A�z�Bڙ�                                    By�'�  T          @��H�}p�?�Q�@P��Bk��B��f�}p�@p�@.{B7{B���                                    By�6�  �          @�  ���ÿW
=@���B�ffCx}q����>���@��HB�aHC0�                                    By�E*  
�          @�Q�>�Q�@333@o\)BS�HB��>�Q�@i��@:=qB�
B��)                                    By�S�  
�          @���?p��@q�@G�A�{B�{?p��@�\)?�  A��RB��                                    By�bv  �          @�33>�  @�\)?ǮA��B��>�  @��>�ff@�\)B��                                     By�q  �          @���>\@�  �&ff��B�>\@N�R�aG��=
=B�ff                                    By��  T          @��;�z�@��R��33�Z�HB�(���z�@�����\��z�B��H                                    By��h  
�          @�(�    @��׿��H���RB�\    @mp��.{��
B�\                                    By��  	�          @�?��@J=q�%��B�L�?��@�H�R�\�K�B��=                                    By���  �          @���<#�
@y���Ǯ���RB�<#�
@k�������B��q                                    By��Z  �          @�����
@w
=?�G�Ac\)B�𤿃�
@\)<�>�G�B�                                    By��   �          @w
=<��
@j�H�����(�B�G�<��
@R�\�������B�33                                    By�צ  "          @vff>W
=@l(����\�v{B�k�>W
=@U������  B��R                                    By��L  
�          @�p�>�@��H�L���2{B�k�>�@qG����
��B��                                    By���  
�          @��R>k�@;��6ff�0��B�  >k�@	���^�R�h��B��q                                    By��  "          @�\)>Ǯ@1��HQ��@\)B�  >Ǯ?�Q��mp��w��B�=q                                    By�>  �          @�����\@7���G����B�8R��\@0�׿L���~{B�                                    By� �  	�          @e��u=#�
@]p�B���C*  �u?^�R@VffB��qBҮ                                    By�/�  �          @mp�?��\@33@&ffB8  B��?��\@6ff?�p�BffB��                                    By�>0  
�          @*=q?^�R@z�?n{A�33B��
?^�R@{>�33@���B�=q                                    By�L�  
(          @%�?���@{?=p�A���B��3?���@�>8Q�@��B��
                                    By�[|  T          @a�?�@QG�<��
>��B��\?�@K��B�\�J�\B��                                    By�j"  
�          @�G�?�G�@P  �����Q�Bvff?�G�@-p��&ff�\)Bd33                                    By�x�  "          @�=q@Q�?xQ��l���\��A��@Q�=�\)�u��hQ�?�33                                    By��n  "          @��\@+������XQ��M�C�xR@+���\)�L���?ffC��                                    By��  "          @��@<(��\(���=q�S��C��@<(�����p  �=�
C�`                                     By���  �          @�p�@
=>�z���=q�zQ�@�ff@
=�J=q�����t�C��)                                    By��`  
�          @��
@>���(�\)AD��@�!G����=qC��\                                    By��  "          @�(��}p�@i���@���
=B�ff�}p�@6ff�qG��MffB�W
                                    By�Ь  �          @��Ϳ�p�@b�\�#33��B�ff��p�@6ff�S33�7{B��)                                    By��R  �          @�ff�Ǯ@�  ��33��  B�W
�Ǯ@l���5���B�Ǯ                                    By���            @�33���
@xQ�����B�ff���
@N{�P  �,�B�B�                                    By���  
�          @�(����@o\)�7
=��RB��ÿ��@?\)�i���B�HB��H                                    By�D  �          @��H�L��@G��e�@z�B��ÿL��@p����R�s  B��                                    By��  �          @�p���{@P  �`���6Q�Bي=��{@������gG�B�                                    By�(�  	�          @�{����@K��a��6=qB�
=����@33����e=qB��f                                    By�76  �          @�\)��G�@`  �Fff���B�R��G�@,���s�
�K��B�W
                                    By�E�  "          @��ÿ�Q�@*�H��Q��Sz�B���Q�?ٙ���Q��~�HC0�                                    By�T�  "          @�����\)@8���r�\�CQ�B� ��\)?�(����H�nffCxR                                    By�c(  "          @�=q����@"�\��\)�a��B�����?��
��{��C 33                                    By�q�  
�          @��ÿ5@(����H�np�B�=q�5?�z�����\)B�{                                    By��t  �          @��׿fff@����
�r�HB�(��fff?�  ����.B��{                                    By��  "          @�Q쿏\)?�(���ff�|{B�p���\)?n{����  CL�                                    By���  "          @�ff�h��?����
=ǮB�z�h��?Y�������RC�{                                    By��f  
�          @�(��W
=?�G���
=�fB�녿W
=?:�H�����C
�
                                    By��  �          @�
=�J=q?�p�����L�B�33�J=q>����
ffC�
                                    By�ɲ  
�          @�33�\(�?�Q���G�u�B�aH�\(�>�����W
C!H                                    By��X  T          @�{�aG�?���z�G�B���aG�>\���H�C�                                    By���  "          @��
��33?��
��\)\B�\)��33���
���H«�qC7�                                    By���  
�          @�p��   ?����\£\C(��   �
=q��=q¢��Cc0�                                    By�J  T          @�\)>�?�����
=W
B��q>�>�����¥��B"�R                                    By��  "          @�\)?��?�����{�)B��3?��?����¢aHB0��                                    By�!�  �          @��?n{?�=q��
=ffBE
=?n{=#�
���H=q@.�R                                    By�0<  T          @��?��?�R�����B7�
?����
=���\¢ǮC�B�                                    By�>�  
�          @���>u?5��¡�RB�G�>u������
=ª�C���                                    By�M�  �          @�p�?��?�R��\)�
A��?����33��Q���C�.                                    By�\.  �          @�
=���?����aHBѮ���?�����¢��B���                                    By�j�  
�          @���W
=?������HaHB��
�W
==��
���R®��CxR                                    By�yz  �          @�\)��Q�?��
��Q��BӅ��Q�>��R��¨ffC��                                    By��   "          @�  >\?}p����H�B��)>\<#�
��ffª�3?�{                                    By���  �          @��R>aG�?^�R���H�HB�#�>aG���G���p�­�C�g�                                    By��l  �          @�?���=u��z�ff@+�?����\(����8RC�C�                                    By��  �          @�ff?ٙ���\���H�\C��{?ٙ���33��z��x{C�@                                     By�¸  
�          @��H?�p��������qC�33?�p���33���H�~
=C�Q�                                    By��^  �          @�(�?p�׽�G���
=.C���?p�׿��\����C�@                                     By��  �          @���>�?
=q��
= �{B@(�>���{���¤33C��3                                    By��  "          @�p�>\)=#�
���
°�Al��>\)�L����G�33C���                                    By��P  �          @�=q?
=q������Q�¦(�C���?
=q��G�����C�H                                    By��  "          @��?c�
>8Q���p�u�A7�?c�
�+����
��C��{                                    By��  
Z          @�  ?�
=?E�����p�B�
?�
=�#�
���\��C�<)                                    By�)B  
�          @���?��H?Y����(���B?��H<#�
��
=
=>�=q                                    By�7�  �          @���?�33>#�
��  G�@�z�?�33�&ff�|��k�C���                                    By�F�  	�          @�ff@z�>�{�q��h�
A
=@z��
=�q��h�C��=                                    By�U4  T          @�Q�?�=q�J=q��=qu�C��?�=q��=q�u��v�
C�
=                                    By�c�  �          @�  >��H��������C�\)>��H�������{C�=q                                    By�r�  �          @�{?�
=�
=q�S33�B��C��H?�
=�1G��3�
�
=C�p�                                    By��&  "          @���@1녿�\)�<(��-��C�� @1녿��'
=��C���                                    By���  "          @�=q?xQ��p��h���u��C��H?xQ��=q�O\)�M  C��{                                    By��r  
�          @��H���
����y���C�녾��
�!G��^�R�Wp�C�]q                                    By��  
�          @��R=L�Ϳ���y���HC���=L���G��aG��e  C��                                    By���  �          @�녾\��(��u=qC|�
�\��Q��a��s(�C�l�                                    By��d  �          @���?��>�{�o\){A*�\?���Ǯ�n�R��C���                                    By��
  
�          @��
@   ?���mp��U�A�G�@   >����w
=�b�
A=q                                    By��  
�          @�(�@p�?���u�aQ�A�Q�@p�?���Q��r  AX��                                    By��V  T          @���@
�H?�z��g
=�M
=B%Q�@
�H?�Q��z=q�fp�A�p�                                    By��  
�          @�33@!G�?˅�\���Dp�B
=@!G�?h���l(��W��A���                                    By��  	�          @��H@!G�?�\)�|(��[�A�\)@!G�>�33���\�h
=A                                       By�"H  
Z          @�?�?��R���\�a��B=  ?�?�
=��(��}��B(�                                    By�0�  
�          @��
@(�?fff���R�rQ�A�Q�@(�=�Q������{@{                                    By�?�  
�          @�Q�?��
?@  ��G�ǮA��?��
�#�
���\C���                                    By�N:  
�          @���?�Q�@%��_\)�=33BT{?�Q�?���z=q�]�B0p�                                    By�\�  �          @��H?���@�\�����a��Bk��?���?�  ���
B=��                                    By�k�  �          @�z�?!G�?�=q�����Bo�?!G�>����¢� A���                                    By�z,  �          @I����(�>���Dz��C��(��.{�E¢��CH��                                    By���  �          @Q녿���\)���h�C?0����333�\)�Z��CNW
                                    By��x  �          @�=q@'
=@g������\BX�R@'
=@P  ��
�癚BL�                                    By��  �          @�@�
@+��   �{BD�R@�
@��<���+ffB-�                                    By���  	�          @�{@p�@0���8Q��B@�
@p�@(��U��5Q�B&p�                                    By��j  �          @��@z�@&ff�L���*�BA{@z�?�p��g��GQ�B"                                      By��  �          @�
=@{@��U�5�RB>z�@{?���n{�R=qB��                                    By��  �          @�?���@�
�n�R�U�B@=q?���?�{��G��qp�B(�                                    By��\  �          @��H?�z�@(��L���B�Bp(�?�z�?�=q�e�d��BQ��                                    By��  �          @�ff?Ǯ@,(��u�K��Bo(�?Ǯ?��H��Q��m�HBM                                    By��  �          @��?W
=@   ���zz�B���?W
=?�p���
=B�B_
=                                    By�N  �          @���>#�
���H°  B�����(������£�{C�y�                                    By�)�  �          @��G���Q�����.CL� �G�������z�ǮCj�3                                    By�8�  �          @��R?��H?���{��q=qBe
=?��H?���ffL�B/                                    By�G@  �          @�33?��R@��q��ap�BW(�?��R?����=q�ffB((�                                    By�U�  �          @�zᾏ\)?����33�HB̊=��\)?�����¤��B�\)                                    By�d�  �          @��
���
@   ����z�B�8R���
?��H��  BѮ                                    By�s2  �          @�{>�G�@�������v�
B�Ǯ>�G�?��R��33�3B���                                    By���  T          @qG�?�R?���X���3B�  ?�R?J=q�e�BOQ�                                    By��~  "          @��ͼ��
?�=q��  ��B�k����
?�\)��  u�B�W
                                    By��$  �          @��>���@���=q�q��B���>���?�z���p�33B���                                    By���  �          @���?c�
@/\)��(��\z�B���?c�
@   ����z�B���                                    By��p  �          @���?��@333�~�R�P�B���?��@�����sG�Bip�                                    By��  �          @�G�?��@U�`  �1z�B���?��@,����Q��U{B�
=                                    By�ټ  �          @�G�?�z�@l(��J�H�(�B�?�z�@G
=�o\)�A��B���                                    By��b  �          @��R>�@fff�U��*
=B��
>�@@  �xQ��P
=B���                                    By��  �          @�  ?B�\@c33�Z=q�,�HB��?B�\@;��|���R�B�                                    By��  �          @�\)>�@H���[��<�B��{>�@!��y���b=qB�u�                                    By�T  �          @�{?   @AG��K��8Q�B�B�?   @���hQ��]��B�
=                                    By�"�  �          @�{�333@(��S33�Q�RB�33�333?�\)�j=q�u��B�\)                                    By�1�  �          @�{�+�?�
=�k���B߀ �+�?�ff�z=q��B��f                                    By�@F  �          @��ÿ�33@�
�{��l=qB�{��33?�����R��C��                                    By�N�  
�          @�G���33@
=q�����k33B�𤿓33?�p����HW
B��3                                    By�]�  �          @��H���
?�Q���p��s��B��쿣�
?��R��8RC�f                                    By�l8  �          @��
�5@7��Tz��Ap�BϽq�5@33�o\)�e(�B�L�                                    By�z�  �          @��\�J=q?���y���y�B�aH�J=q?�p�����B�B�(�                                    By���  �          @��
�E�@0���7��4
=B�LͿE�@G��QG��W(�Bٮ                                    By��*  �          @��?�p�@���ff�o�
B�� ?�p�@���G����\B���                                    By���  �          @�33?�33@��\����ȣ�B��{?�33@�����p(�B���                                    By��v  �          @��H?z�H@��H�����I��B�8R?z�H@���������
B�{                                    By��  �          @�=q?}p�@�
=?}p�AC
=B��?}p�@�=q>�\)@W�B�u�                                    By���  �          @�p�?�z�@�\)?У�A�=qB�� ?�z�@�?^�RAz�B��\                                    By��h  �          @���?�=q@��\?޸RA��RB���?�=q@���?uA)p�B��
                                    By��  �          @��\?���@�?O\)AB�  ?���@�  <�>�p�B�p�                                    By���  
�          @�33?��@�
=?�ffA�{B��?��@���?E�A��B��q                                    By�Z  T          @���?���@�Q�>�  @.�RB���?���@�������\B���                                    By�   �          @���?Ǯ@��>��@�p�B��R?Ǯ@�=q��p���  B��q                                    By�*�  �          @�ff?�z�@��
?��A8��B��?�z�@�\)>�z�@EB��3                                    By�9L  �          @�\)?��R@�z�?��A?�
B�?��R@�Q�>�{@e�B�k�                                    By�G�  �          @�  ?�  @���@  ��p�B�\?�  @�녿��
��
=B�(�                                    By�V�  �          @��?�\)@�ff��(���{B���?�\)@�(��(���B�W
                                    By�e>  �          @��?�
=@�p��
=q��(�B�\)?�
=@�G��5�����B�.                                    By�s�  �          @��?�  @�\)�
=��
=B�p�?�  @��\�@  �\)B���                                    By���  �          @���?�Q�@@�����R�NG�B��?�Q�@ff����lp�Bj(�                                    By��0  �          @�z�@��@O\)�k��,�HB^�@��@)�����
�H33BI(�                                    By���  �          @��?�{@k��tz��1�B�B�?�{@Dz���=q�P��B�{                                    By��|  �          @�=q?�(�@p  �p���)�\B��?�(�@I�������G��Bu�\                                    By��"  T          @�{@�R@�33��̸RB{Q�@�R@}p��=p��
=Br\)                                    By���  �          @�p�@�R@���{���B|�\@�R@�G��5���HBt=q                                    By��n  �          @��@<(�@�G����R��Q�B^\)@<(�@����ff���HBW�                                    By��  �          @��
@8Q�@�������d(�Bg��@8Q�@������{Bb                                      By���  �          @�z�@>{@��׿������Bb�H@>{@�  �p���z�B\=q                                    By�`  �          @�@>{@��R������=qBa\)@>{@����{�Σ�BY��                                    By�  �          @��R@R�\@���������BNp�@R�\@s33�-p����BD��                                    By�#�  �          @���@`  @`  �:=q��\)B4�@`  @C33�XQ��{B$p�                                    By�2R  �          @�G�@K�@Z�H�Y���=qB<Q�@K�@:=q�vff�'33B)��                                    By�@�  �          @��R@*�H@`���g
=�{BSG�@*�H@>{��=q�6�B@\)                                    By�O�  �          @��@-p�@?\)���
�6=qB>�@-p�@Q�����L��B$�H                                    By�^D  �          @�@
�H@'���Q��S�RBIff@
�H?�(����\�k
=B(�                                    By�l�  T          @�Q�>�{@��Ϳu�!G�B���>�{@�
=���H����B��{                                    By�{�  �          @�  ?333@��
������p�B�p�?333@���"�\�ܣ�B�u�                                    By��6  �          @���?�33@��������Q�B�ff?�33@�{�;���(�B��                                     By���  �          @��?��@��R�-p�����B���?��@�G��U�33B�8R                                    By���  �          @��H?L��@�p�����\)B�u�?L��@���'
=�ߙ�B�\)                                    By��(  �          @�33?.{@�  �P  �  B��q?.{@�Q��u��-  B��3                                    By���  �          @��>�33@qG��\)�9��B�u�>�33@L(����R�Y{B��\                                    By��t  �          @���?333@��
�޸R����B��=?333@��H�Q���33B��\                                    By��  �          @�G�?��
@�=q��=q�5�B��H?��
@���}p��%B�u�                                    By���  �          @��@   @�ff�G���ffB��f@   @�z��(Q����HB���                                    By��f  T          @�
=>��@�\)�*�H���B��q>��@��\�Q���B��R                                    By�  �          @�\)?�
=@����E��p�B�Q�?�
=@tz��hQ��!{B|�R                                    By��  �          @���?�\)@z=q�h��� 33B���?�\)@X����(��;\)Bt                                    By�+X  �          @�ff@O\)@�Q��
=���BS=q@O\)@�  ��R���BL=q                                    By�9�  �          @�ff@I��@��H�Ǯ����BX33@I��@�33����{BQ�H                                    By�H�  �          @�G�?E�@XQ������U33B�8R?E�@/\)�����rffB�z�                                    By�WJ  �          @�
=?
=@N{��{�\\)B�#�?
=@%�����yB��                                    By�e�  �          @�{>Ǯ@'������x�B�  >Ǯ?�Q����\B�B�                                    By�t�  �          @�<�?�ff���

=B�
=<�?������fB�                                    By��<  �          @�p���p�?�z������B���p�>�G����H¨(�C33                                    By���  �          @��\�=p���\)��{£ffCH��=p��xQ����=qCh��                                    By���  T          @��
�8Q��ff����¢��CS�)�8Q쿓33����Cn                                      By��.  T          @�����\�*=q���
�X=qCh����\�N�R��  �@33Cm��                                    By���  �          @�=q�����+���(��f{Co�������Q���Q��LQ�Cts3                                    By��z  �          @���
�H��\)��{�t�CX�
�H��R��{�a�RCa�=                                    By��   �          @��R�_\)���
���R�O�HC5O\�_\)�#�
��p��L�C>ff                                    By���  �          @����
=�#�
��� C5�ῷ
=�8Q���(��)CN�{                                    By��l  �          @�{��>���z���C"=q���aG�����#�C<                                    By�  �          @�Q�?Q�@����Q���B��?Q�?��
����)Bw=q                                    By��  �          @��?�{@	����p��)Bz��?�{?��R������BU�\                                    By�$^  �          @�Q�?+�@!����
�{��B�u�?+�?����z�\)B��                                    By�3  �          @�(�?�\@N{���\�Z33B��?�\@(����{�u��B�8R                                    By�A�  �          @��
�Ǯ@l(���ff�B{B�(��Ǯ@I�����
�]��B�=q                                    By�PP  �          @��\��\)@A���(��c=qB�Ǯ��\)@(���
=�33B�p�                                    By�^�  T          @��=��
@333����pG�B�=��
@(����\B��)                                    By�m�  �          @�(�>��
@   �����}�\B��{>��
?�\)��G�� B��)                                    By�|B  �          @��?z�H@*�H����l(�B��?z�H@������
B��=                                    By���  �          @��H?�p�@u�hQ��"�B�ff?�p�@XQ�����;�RB{G�                                    By���  �          @�Q�?
=q@Z=q��{�J�B���?
=q@8Q���=q�e(�B�                                    By��4  �          @��
>\)@\�����\�M�
B�L�>\)@:=q���R�i{B�p�                                    By���  �          @�Q��ff@5����j{B�#׾�ff@�����p�Bʣ�                                    By�ŀ  �          @�G��aG�<��
��33¡G�C2z�aG��(���=q8RCV��                                    By��&  �          @�(���(�?   �����{C#�3��(������\
=C8
=                                    By���  �          @��
���?(����(�{C�=���<��
��p�33C3�                                    By��r  �          @�z�W
=?���G���C�{�W
=��G����¢��C;��                                    By�   �          @��R�:�H?8Q����G�CW
�:�H=�\)����¥=qC.�H                                    By��  �          @����(�?�������W
Cp���(�?W
=����p�C+�                                    By�d  �          @�p���?��R��  �z  C  ��?�33��{ffC�R                                    By�,
  �          @�  �������ff©�CP33���h����(�W
Cr5�                                    By�:�  �          @�{�J=q?��R��8RB�8R�J=q?#�
��G�#�C�                                    By�IV  �          @�\)�h��@R�\����O��Bҽq�h��@1���ff�h��B��                                    By�W�  T          @�(�����@x����  �6�B��3����@Z=q����P�
B��                                    By�f�  �          @�p��8Q�@J=q���\�Z�\B�Ǯ�8Q�@(Q������s�RB���                                    By�uH  �          @�=q�8Q�@����z��B�Ǯ�8Q�?\��33\)B��q                                    By���  �          @����@p���z�  B�\��?�����33Q�Bۅ                                    By���  �          @�����@���p�ǮBӳ3���?�  ���
�B߮                                    By��:  �          @��H��R?�
=���
�fB���R?L����  ��B��H                                    By���  �          @��Ϳ8Q�?�{��  B�G��8Q�>�����H£.C33                                    By���  �          @���Tz�?�\)��p�u�B�=�Tz�?:�H����L�C
��                                    By��,  �          @�����\?�\)����L�B�Q쿂�\?=p�����G�C�3                                    By���  �          @�(���(�?fff���H��C����(�>�z�����B�C)�                                    By��x  �          @����
=@~�R�W��ffB�Q��
=@e��q��%�B���                                    By��  �          @�\)��\@s�
�p  �#�RB�=q��\@XQ���z��9p�B��                                    By��  T          @�ff�(�@fff�\(��z�B�Q��(�@Mp��s�
�-�B���                                    By�j  
�          @�����@xQ��K���\B�G����@`���e�� \)B�p�                                    By�%  �          @����@�G���p���\)B�{�@����p�����B�#�                                    By�3�  �          @��R�(��@�(���=q����B�
=�(��@����\��p�B��                                    By�B\  �          @����E�@�=q�E����B�8R�E�@}p������V�RB���                                    By�Q  �          @�ff��@�G��)����
=B�Q��@~�R�E���B�\                                    By�_�  �          @��׿�\)@���C33�Q�B�z῏\)@���`  ��
B���                                    By�nN  �          @�녿(�@��H���H��
=B�p��(�@�����H��
=B���                                    By�|�  �          @��H��R@�Q����\)B�aH��R@���G��/�B��{                                    By���  �          @�G����R@��ÿ^�R�z�BϏ\���R@�p�����uB�33                                    By��@  �          @�\)�(�@[��}p��1
=B���(�@?\)�����D�
B�G�                                    By���  �          @����z�@.�R��=q�I�CY��z�@G����\�Z��C�f                                    By���  �          @�����z�@{�����tp�B�ff��z�?�Q���
=G�Ck�                                    By��2  �          @�\)��  @ff����u{C���  ?�=q���C

=                                    By���  �          @�(���
�B�\���\��C8����
�=p���G���CE�3                                    By��~  �          @�33�0��?�{��  �[�RC�q�0��?�\)����g��C��                                    By��$  �          @���P��@S�
�p  ���C���P��@:=q����+C
=q                                    By� �  �          @�{�C33@g��aG��z�C{�C33@P  �w��#�C33                                    By�p  �          @�
=�p�@2�\���R�H�Cff�p�@���
=�YG�C�H                                    By�  �          @��;�@�H��p��E�Cz��;�?��H��z��S  C:�                                    By�,�  �          @�
=�\)@*=q�e���CL��\)@�\�u��{C.                                    By�;b  �          @���u�@.�R�s33��C�=�u�@������(ffC�f                                    By�J  �          @�(��l��@J=q�k��ffCxR�l��@1��~�R�"C
                                    By�X�  �          @Å�tz�@=p��mp���CB��tz�@$z��\)�#�C�                                    By�gT  "          @��H����@.{�C�
��
=C������@���S�
�
z�C�q                                    By�u�  �          @��\��=q@'
=�O\)�Q�CO\��=q@��^�R�
=C                                    By���  T          @�����@(Q������C�f��@������=qC��                                    By��F  
�          @�z����R@)����p����HCٚ���R@{��p�����C�                                    By���  �          @�����(�@�׿�(��j�RC@ ��(�@ff��
=����C��                                    By���  "          @��H���@�R��p���Q�Cn���@33��Q����C&f                                    By��8  
�          @�����R@(�ÿ����yp�C�R���R@{������33Cz�                                    By���  T          @������\@U��G��!p�C�)���\@N�R��=q�Up�C��                                    By�܄  T          @�z����@s33�Tz���C=q���@mp������;�C��                                    By��*  "          @�  ���@x�ÿ=p����
C�����@s�
��{�2=qC	�\                                    By���  �          @������
@q녿333���HC	u����
@l�Ϳ���,��C
)                                    By�v  T          @��\��p�@i���L�Ϳ\)C
����p�@hQ������C
�q                                    By�  �          @�Q�����@n{>8Q�?�C	k�����@n{�8Q��\)C	k�                                    By�%�  �          @�(����@\�Ϳ�\��Q�C�����@X�ÿW
=�
{C#�                                    By�4h  
�          @����
=@XQ�L���=qC޸��
=@R�\��\)�6ffC��                                    By�C  �          @�  ��z�@O\)�u��\C
��z�@HQ쿢�\�K�
C��                                    By�Q�  �          @���w
=@p�׿!G���ffC���w
=@l(��}p��'\)CE                                    By�`Z  
Z          @��
�l(�@�
=�Ǯ�~�RC(��l(�@�p��J=q� z�C��                                    By�o   	�          @��
�L(�@�=q>�(�@���B��q�L(�@��H    ���
B��                                     By�}�  "          @�G��AG�@��\?@  @���B����AG�@�(�>��
@S�
B�=q                                    By��L  
Z          @�{�\)@z=q�s33��C�
�\)@s33����Up�C^�                                    By���  	�          @����hQ�@�=q�}p��
=C  �hQ�@��R����`(�C��                                    By���  T          @�{���@�33���\�F�HC�R���@}p���33��p�C��                                    By��>  "          @�G��p��@����\)��=qC
=�p��@q��6ff���HCٚ                                    By���  �          @�p��c�
@l(��5��=qC�3�c�
@Z=q�J�H�{C33                                    By�Պ  
(          @��
�dz�@&ff�w
=�$��C�)�dz�@\)���\�0z�C��                                    By��0  
�          @��H�|(�@��fff�=qC���|(�?����r�\�#33C�R                                    By���  
�          @�����\)@*=q�8����=qC޸��\)@Q��G���HC��                                    By�|  
�          @���tz�@K��<(�����C0��tz�@9���N{�p�CǮ                                    By�"  �          @����Z�H@e��,(���  C���Z�H@Tz��@  � �
C�\                                    By��  T          @���:�H@��\�1����B�G��:�H@s�
�H����B�                                    By�-n  
�          @��\�>�R@�
=�.�R����B�u��>�R@}p��Fff� 33B�                                      By�<  
�          @�=q�?\)@�ff�33���B����?\)@�
=�,(��ۮB��{                                    By�J�  �          @��\�=p�@�ff��\���
B�(��=p�@�\)�,(���  B��                                    By�Y`  
Z          @��H�7�@��\�z���G�B�(��7�@�(��{��  B��=                                    By�h  
�          @�(��5�@����   ���HB��5�@����9����B��H                                    By�v�  �          @�{�N�R@�
=��R��{B����N�R@�Q��'���
=B�W
                                    By��R  
�          @�\)�.{@���E���
B�L��.{@|���\�����B�\                                    By���  
�          @���(�@k������,��B�\�(�@S�
���\�=�B�\                                    By���  �          @�p����@z������B��;��?�=q���H��BШ�                                    By��D  �          @�{�#�
@2�\��{�tz�B��f�#�
@z���p�{B��                                    By���  �          @������@j=q��ff�5�\B�=q����@QG���Q��G33B�3                                    By�ΐ  "          @��
���@^�R����E��B�33���@E������X�B�{                                    By��6  
�          @�=q�(��@u��G��>��B�p��(��@\������R��Bə�                                    By���  T          @����R@`  ��Q��G\)B�
=���R@E�����Y�RB�aH                                    By���  �          @��R���\@���\)=qB�
=���\?˅��z�  B�\                                    By�	(  �          @���(�@!G���\)�j�B��{��(�@z����z(�C�H                                    By��  
�          @�33�\)@`����G��0�B�
=�\)@H�����\�A=qB��)                                    By�&t  T          @���#�
@7
=����Bz�C޸�#�
@{���H�P�
C\                                    By�5  
�          @����G�@����\�FQ�C8R�G�?�
=��Q��Q�CL�                                    By�C�  "          @ƸR��?�ff���u�C���?�����=q  C��                                    By�Rf  
�          @�G��
=q?�p������{Q�C^��
=q?�G�����G�C�)                                    By�a  "          @�  �7
=?�z���z��[p�CG��7
=?�p���G��e�C�H                                    By�o�  T          @���Mp�@�����>=qC���Mp�?�33����I  Cc�                                    By�~X  �          @�\)�k�@0���W����C+��k�@p��fff�\)CE                                    By���  
�          @��R��
=@Mp��33��
=C�\��
=@AG��z���ffCu�                                    By���  �          @�p��z�H@`���33���C
!H�z�H@Tz����C��                                    By��J  �          @�  ���@hQ쿑��8(�Cn���@aG���Q��j�RCL�                                    By���  �          @�����G�@h�ÿ�G��I�C����G�@aG��Ǯ�{\)C��                                    By�ǖ  �          @����r�\@n{�33����C}q�r�\@b�\�
=���
C�R                                    By��<  "          @���~{@y����G��rffC���~{@p�׿�=q��=qC��                                    By���  �          @�\)���\@g
=?(�@�
=C#����\@i��>��R@EC�
                                    By��  T          @�
=��@,(�@.{A�C5���@:=q@�RA�{C�                                    By�.  
�          @�\)��z�@-p�?���A1�C����z�@333?^�RA�C=q                                    By��  �          @����
@�H?���A;�C�\���
@ ��?k�A��C�                                     By�z  �          @�����@5@�Aʣ�C^����@A�@�A�(�C��                                    By�.   �          @�(����
@>{?˅A�  Cٚ���
@E?��A`  C�                                     By�<�  �          @�ff����@HQ�?�  A!C)����@L��?:�H@���C�                                     By�Kl  �          @��R��@e�>aG�@�RC
��@e���
�L��C
=                                    By�Z  �          @�(��q�@��
��G���  C���q�@�=q�G�� ��C��                                    By�h�  "          @�p����@e>W
=@
=C}q���@fff��Q�s33Cu�                                    By�w^  �          @�33�\)@y��<��
>�  C���\)@x�þ����FffC�                                    By��  �          @���b�\@����W
=�ffC+��b�\@~{��
=�G�
C�                                    By���  "          @��\�p��@~{�s33�{Ck��p��@xQ쿣�
�UG�C�                                    By��P  
�          @����I��@����p����B�ff�I��@��R����\��B���                                    By���  T          @��H�-p�@���k����B�u��-p�@��Ϳ���Z�RB�p�                                    By���  
�          @�  �AG�@��׿!G����B�ff�AG�@�ff��  �(Q�B�33                                    By��B  T          @�=q�P��@�33�����1p�B����P��@�  ���n=qB���                                    By���  "          @�G��qG�@q녿���\  C��qG�@j=q��{���RC�)                                    By��  �          @�G���\@�Q����{B��)��\@��R�^�R��RB�B�                                    By��4  �          @��ÿ��R@�  ��\��=qBӣ׿��R@�{�p���{B�                                      By�	�  �          @����(�@����O\)�Q�B���(�@�
=��  �Lz�B؀                                     By��  �          @�����  @���\)����Bي=��  @�33�}p��&{B���                                    By�'&  	�          @��H�   @�Q쿢�\�T(�B߀ �   @�(���
=��33B�z�                                    By�5�  T          @�33�'�@�G���  ���
B��f�'�@�(�����z�B�3                                    By�Dr  "          @�\)?�ff@�Q�?���A�p�B�B�?�ff@��
?���A9p�B���                                    By�S  �          @�G�?��@���@��Ař�B��?��@�\)?�p�A�
=B��f                                    By�a�  "          @�?��@��@^�RB�\B�\)?��@�Q�@FffB ffB�#�                                    By�pd  �          @�{?�=q@|(�@��B1p�B��q?�=q@�Q�@n�RBz�B�\)                                    By�
  �          @�?˅@�(�@Q�B33B�#�?˅@�(�@8Q�A��B���                                    By���  �          @��
?�z�@���@G�B33B�Ǯ?�z�@�z�@.�RA��HB��\                                    By��V  T          @��
?�Q�@�\)?�z�A�  B�p�?�Q�@�33?��HA>ffB��                                    By���  "          @�z�?z�H@�Q�?�{A�z�B���?z�H@�(�?�33A6�RB�33                                    By���  T          @�p���Q�@���@ffA�{B�uþ�Q�@�p�?��A��\B�B�                                    By��H  T          @�(��#�
@�
=@33A�=qB�녽#�
@��
?˅A}G�B��f                                    By���  �          @���@�\@��\>�G�@��B�#�@�\@�33    <��
B�B�                                    By��  
�          @��
?�\)@�Q�@I��BQ�B��=?�\)@�  @2�\A�(�B��
                                    By��:  �          @��?��H@��R?�A�ffB�.?��H@��H?�33Al��B���                                    By��  T          @�  @U�@��
��\)�:�HBX�
@U�@�33��ff��G�BXQ�                                    By��  T          @���?���@���>8Q�?�B�p�?���@��׾�  �#�
B�k�                                    By� ,  T          @�\)?�{@��R>W
=@��B��?�{@��R�k��
=B��                                    By�.�  �          @�Q�?(��@��H    <#�
B���?(��@�=q������B��{                                    By�=x  T          @��\�aG�@�
=�h����B�aH�aG�@��
�����\  B�u�                                    By�L  T          @�=q�#�
@��R�fff�B��
�#�
@��
��{�XQ�B�
=                                    By�Z�  �          @�=q��R@�p������*=qB��=��R@�=q���
�t��B�Ǯ                                    By�ij  �          @���>L��@�p������+33B��>L��@�녿��
�v{B��)                                    By�x  T          @�Q�?fff@��H��ff��  B��?fff@�p��p����HB�G�                                    By���  T          @��>W
=@����z����B��>W
=@�\)���
�*�HB���                                    By��\  �          @��H<�@��R�(���B�Q�<�@���5��G�B�G�                                    By��  �          @�33?:�H@����z����B��?:�H@������=qB�\)                                    By���  �          @�G�?
=q@�z�����
=B���?
=q@����
��ffB��{                                    By��N  
�          @�Q����@��?#�
@�=qB������@�33>L��@��B�z�                                    By���  T          @��H<��
@�\)?B�\@�B��=<��
@���>��R@O\)B��=                                    By�ޚ  �          @��R<#�
@�>�33@c33B��H<#�
@�{��G���33B��H                                    By��@  T          @�ff�L��@��>�p�@o\)B���L��@�p���Q�s33B��                                    By���  �          @���'
=@��
���\�'�B�k��'
=@��׿��iB�ff                                    By�
�  �          @�p��G�@����\��{B�z��G�@�33�n{��
B���                                    By�2  �          @������@�  ����&ffB�aH���@��Ϳ�(��l  B�.                                    By�'�  T          @��R�G�@�Q쿴z��f=qB����G�@�(�������\)B���                                    By�6~  �          @��7�@�{�p�����B����7�@~{�333��B��                                    By�E$  �          @�  ���@����
=����B�k����@��\�   ��Q�B�Q�                                    By�S�  �          @��%@��H�ff���
B���%@����{��z�B�                                    By�bp  �          @����H@��׿�ff����B��)��H@���(���Q�B�                                     By�q  �          @���P��@��?�\@��RB��
�P��@���>�?���B��                                    By��  �          @���Vff@�z�?�@��B���Vff@�p�>.{?�p�B�Q�                                    By��b  �          @����	��@�ff�:�H��B�#��	��@��
��\)�EB��)                                    By��  �          @�33�z�H@�p���\)��33B�p��z�H@����G���=qB��                                    By���  �          @�{���
@�ff����G\)B�p����
@��\��ff��B�.                                    By��T  �          @�33�3�
@��?�  A,��B�(��3�
@�
=?!G�@�G�B�\)                                    By���  �          @����@��ÿ�  �6�RB�G���@���\)�{�B�=q                                    By�נ  �          @�ff��p�@��	����{B����p�@��� ����\B�u�                                    By��F  �          @����%@�\)�0����(�B�
=�%@������;�B��)                                    By���  �          @���   @�
=��\)�Tz�B�z��   @�ff��G���
=B�R                                    By��  �          @��\��@�p��=p��
=B�aH��@��H��\)�N�RB���                                    By�8  �          @��O\)@i��?�z�A�
=C�\�O\)@p��?���AH��C�                                     By� �  �          @�
=�u@C33?��HA�  C�{�u@K�?���A�  Cc�                                    By�/�  �          @����dz�@E�?��
A�C0��dz�@N{?\A�z�C	�                                    By�>*  �          @��H�J=q@dz�?ٙ�A��RC���J=q@l��?��A�C�                                    By�L�  �          @����@  @fff?�=qA���CǮ�@  @o\)?��
A�33C ��                                    By�[v  �          @�\)�l��@j=q?ǮA���CQ��l��@q�?�  AT��Ck�                                    By�j  "          @�Q�����@J=q?�Q�A�p�C�q����@R�\?�Ao�C��                                    By�x�  "          @����q�@e�?�\)A�z�C���q�@n{?���A��
Cn                                    By��h  �          @���a�@`��@33A�(�C��a�@l(�?��RA���C��                                    By��  
�          @�
=�Z=q@W
=@'�A�33Ck��Z=q@dz�@�A�C�3                                    By���  �          @��J�H@;�@=p�B	�
C	=q�J�H@J�H@-p�A���C                                      By��Z  T          @���G
=@%@;�Bz�C.�G
=@5�@,��BC	�3                                    By��   �          @��׿#�
@Fff���G�B�8R�#�
@8���'��&33B��H                                    By�Ц  "          @�33?�?�\���ffAn=q?�>#�
��z�\@��                                    By��L  �          @���@�>������v=q@o\)@��W
=�����v  C�c�                                    By���  �          @��
@(Q�>�  ��33�j�H@�p�@(Q������k\)C���                                    By���  T          @��?�
=?�  ���\�}ffB�?�
=?\(���p�\)A�\)                                    By�>  T          @�z�@ ��?��R����z�\A�33@ ��?Y������A���                                    By��  �          @��@{@z�����^{B,  @{?�
=��=q�j�B�H                                    By�(�  �          @�
=@�@ff��=q�QQ�B2�@�?�(���Q��^z�B�                                    By�70  T          @�G�@
=q?��
��\)�m��B��@
=q?�\)����y�BG�                                    By�E�  �          @�(�@33@ ����p��cG�B$@33?�����=q�offB��                                    By�T|  "          @���@ ��?�Q����H�l�RA��@ ��?��\��ff�u\)A���                                    By�c"  �          @�  @�H?�ff�����p\)Aᙚ@�H?aG�����xz�A�Q�                                    By�q�  �          @�@Q�?c�
��G��xz�A�  @Q�>����33�}�HA4                                      By��n  �          @�33?�z�k����R�{C���?�z�(����\)C���                                    By��  "          @��
?��R�����R{C�/\?��R�h��������C���                                    By���  �          @�(�@p���Q���z��HC�ff@p��G���33k�C�E                                    By��`  �          @�=q@)���8Q������o  C�@)���z������l(�C�ٚ                                    By��  �          @��@<(���R�����[��C�@<(��}p����R�V�C���                                    By�ɬ  �          @��
@:=q���������e{C�
=@:=q���H��  �c
=C�33                                    By��R  �          @��@;��#�
����e��C�ff@;������\�cz�C���                                    By���  �          @�ff@K��8Q���p��X�C�k�@K��\)��z��V��C��q                                    By���  �          @��R@[�=��
��\)�K�R?���@[���z���
=�K�C��{                                    By�D  �          @��R@3�
=u��p��k��?���@3�
��33�����j�\C�g�                                    By��  "          @�G�@?\)>�  ��33�c\)@�@?\)�#�
��33�c��C�w
                                    By�!�  
�          @�33@
=q?�
=����qp�B{@
=q?�  ��{�|A�\)                                    By�06  �          @��\@P��?��
���
�?�HA�R@P��?�z���Q��H��A��\                                    By�>�  "          @�Q�@K�?�z�����>=qA�z�@K�?�ff���R�G��Aϙ�                                    By�M�  
�          @���@N�R?�����  �:��A��\@N�R?�=q����D�\A���                                    By�\(  T          @�G�@S�
@�\�{��5{A���@S�
?�
=���H�?�A�\)                                    By�j�  
�          @��@Q�@������{B~�@Q�@�=q�.{����Byz�                                    By�yt  T          @�=q?���@��
��Q���Q�B�L�?���@�p����ffB��{                                    By��  T          @�G�?�z�@����0  ��Q�B�� ?�z�@r�\�Fff��B|��                                    