CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240718000000_e20240718235959_p20240719021548_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-19T02:15:48.385Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-18T00:00:00.000Z   time_coverage_end         2024-07-18T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy27�  �          @��?�G�@9��@ffA���Bk�?�G�@<��@�A�ffBl��                                    By2F&  "          @�\)?��@e�?���A�z�B�� ?��@g�?�G�A�z�B��                                    By2T�  
�          @�ff?�Q�@e�?���A�\)B�aH?�Q�@g
=?�p�A��B��q                                    By2cr  �          @�Q�?�=q@Z�H?�(�A���B��{?�=q@^{?��A��HB�(�                                    By2r  �          @�\)?�(�@p��?��A}�B�.?�(�@r�\?�ffAhQ�B�u�                                    By2��  �          @���?�\)@s33�#�
��B(�?�\)@s33�\)��=qB�                                    By2�d  "          @�����(�@�@~{Br��B�G���(�@�@z=qBm\)B�W
                                    By2�
  �          @���R@��@n{Bi�HBӀ ��R@�\@j=qBd\)B�G�                                    By2��  (          @�����
@$z�@dz�BX�HB��R���
@*=q@`  BS
=B���                                    By2�V  
�          @��
���@;�@J=qB:=qB�
=���@@��@E�B4z�B�p�                                    By2��  �          @�(��
=@H��@>�RB+��B���
=@N{@9��B&�BȞ�                                    By2آ  �          @��
�G�@=q@w�Bc  B�녿G�@ ��@s33B]G�B֔{                                    By2�H  
�          @���z�@,(�@o\)BV�HB�B��z�@1�@j�HBQ  B�k�                                    By2��  T          @�
=��@c33@,(�B��B����@g�@%Bz�B�{                                    By3�  
/          @�=q���?�\)@�\)B��B�����?޸R@�{B��\B��                                    By3:  
�          @�{�J=q@z�@�z�Bwp�B��J=q@�@��\Bq��B���                                    By3!�  
�          @���:�H@�@�33Bv�RBڏ\�:�H@��@�G�Bp��BظR                                    By30�  "          @���c�
@
=@\)Bp�B���c�
@{@{�Bj��B�Ǯ                                    By3?,  T          @�z�E�@�@�=qBu�\B�ff�E�@��@�Q�Bo�B�z�                                    By3M�  T          @�z�&ff@p�@���Bp��B�Ǯ�&ff@�@|��Bj\)B�B�                                    By3\x  T          @�G��+�@�R@y��BlB�G��+�@@uBfz�B���                                    By3k  �          @�����@��@tz�Be(�Bνq��@   @p  B^�B͙�                                    By3y�  "          @����:�H@G�@�  Bv�
B��:�H@��@|(�Bp�B�                                    By3�j  �          @������@	��@}p�Br\)B�8R���@G�@y��Bk�
Bѳ3                                    By3�  �          @�G��   @
=q@|(�Br�B�33�   @�@w�Bk�
B��f                                    By3��  �          @��׽�@�@u�Bl��B�=q��@��@p��Bf
=B��                                    By3�\  �          @�{?k�@ff@i��B]33B�u�?k�@p�@e�BV�B�33                                    By3�  T          @��?z�H@��@o\)Bd�B���?z�H@�
@j�HB^z�B�{                                    By3Ѩ  �          @���?�33@   @e�BQ�B��?�33@'
=@`  BKffB��                                    By3�N  �          @��?��R@%�@]p�BI�\B��H?��R@,(�@XQ�BC
=B���                                    By3��  T          @�\)?��@�H@`��BN=qBp��?��@!�@[�BG�Bu                                      By3��  T          @�Q�?�(�@5�@L(�B4{Bz\)?�(�@;�@EB-�\B}��                                    By4@  �          @�ff?���@7
=@@��B*33Bs=q?���@<��@:=qB#�RBvff                                    By4�  �          @�ff?У�@8Q�@>�RB(�Bq�R?У�@>{@8��B!��Bt�                                    By4)�  �          @�ff?�
=@1G�@7�B!z�B\�\?�
=@7�@1�BG�B`{                                    By482  "          @�z�?�33@'�@AG�B1{Bgz�?�33@.{@;�B*��Bk=q                                    By4F�  �          @�G�?�=q@O\)@#�
BffB�L�?�=q@U�@��B{B�(�                                    By4U~  "          @�  ?�ff@N{@!�B�B��R?�ff@S33@�HB�RB��{                                    By4d$  �          @�33?��@A�@!G�B=qBm�\?��@G
=@=qBz�BpQ�                                    By4r�  "          @��H@�R@p�@333B&�
B3(�@�R@�
@.{B!33B8                                      By4�p  T          @���@
=?�(�@L(�B?�\B,=q@
=@�@G�B:  B2z�                                    By4�  �          @��?�@��@FffB<�BH�?�@�
@AG�B6��BM�H                                    By4��  �          @�z�?z�@o\)?�p�A�
=B�Q�?z�@r�\?���A��HB��=                                    By4�b  �          @�(�>�=q@z=q?�  A�G�B���>�=q@|��?�{Ay��B�\                                    By4�  T          @�  ?��>���@x��B��qA��?��?
=q@w�B��A��
                                    By4ʮ  
�          @��;�@p�׾���p�B�\��@o\)����(�B�#�                                    By4�T  T          @��Ϳ:�H@j=q��G���z�Bʨ��:�H@g
=��33���B�                                    By4��  )          @��<�@�(��s33�N=qB�8R<�@��H�����pz�B�8R                                    By4��  
�          @�{�aG�@z�H������ffB�W
�aG�@w�������B�p�                                    By5F  
�          @�\)�(�@��׿�\)�vffB�G��(�@~�R���\��z�B�z�                                    By5�  �          @��s33@{����
�e�B���s33@xQ쿗
=���B�ff                                    By5"�  "          @��k�@s�
�����=qB�
=�k�@p�׿������B�p�                                    By518  T          @����
=@hQ����=qB�{��
=@c33��p���\)B�Ǯ                                    By5?�  [          @�{�0��@j�H�Y���L��B�=q�0��@h�ÿ}p��pQ�B�p�                                    By5N�  
S          @�
=?^�R@fff@   A�B��
?^�R@j�H?�{A��
B�ff                                    By5]*  �          @�{?G�@hQ�?���A�\)B���?G�@mp�?�ffA�33B�G�                                    By5k�  �          @��R?333@^{@�BQ�B�
=?333@c�
@��A�Q�B���                                    By5zv  �          @��R?aG�@dz�@�\A��B�\)?aG�@i��?��Aՙ�B��                                    By5�  "          @�\)?�@fff@��A�G�B���?�@l(�?�p�A�ffB�
=                                    By5��  
�          @���>�(�@e�@�
Bp�B�L�>�(�@j�H@
=qA�B���                                    By5�h  M          @��>�{@q�@�\A�z�B���>�{@w
=?��A��B�
=                                    By5�  
�          @���>�(�@tz�?���A�  B�.>�(�@x��?�Q�A��\B�p�                                    By5ô  �          @�G�?��@{�?У�A�Q�B�#�?��@\)?���A���B�ff                                    By5�Z  T          @���?��@o\)?�A׮B�aH?��@tz�?�G�A��B��R                                    By5�   T          @�{?u@I��@$z�B�B�#�?u@P��@�BG�B�8R                                    By5�  
�          @���?�R@\��@p�B ��B���?�R@b�\@33A홚B�.                                    By5�L  
�          @�>�G�@k�?��A�Q�B�z�>�G�@p��?�(�A��B�                                    By6�  "          @���>�\)@|��?�=qA�B���>�\)@���?�33A��B��                                    By6�  
�          @�\)��z�@{�?   @�=qBۅ��z�@|��>��
@��B�\)                                    By6*>  T          @�p�?�@dz�@#�
Bp�B���?�@k�@��B
=B�\)                                    By68�  �          @��?��@c33@��B�RB���?��@j=q@�A�z�B��=                                    By6G�  
�          @�33>��@X��@*�HB�B�W
>��@aG�@ ��B
=B��H                                    By6V0  [          @�33>Ǯ@_\)@#33B�HB�ff>Ǯ@g
=@Q�B�B���                                    By6d�            @�33>�ff@^{@%B�B�#�>�ff@e@�HBG�B���                                    By6s|  
�          @���>��
@l(�@
=B��B���>��
@s33@�A�  B�B�                                    By6�"  
�          @�33>\)@fff@�HB��B���>\)@n{@  A�
=B�Ǯ                                    By6��  
�          @��\>��@j=q@�B�B�aH>��@qG�@	��A�p�B��                                    By6�n  M          @���<��
@c33@Q�BG�B�k�<��
@j=q@��A��B�p�                                    By6�  
�          @��>#�
@Tz�@'�B�B�k�>#�
@\��@p�B�RB���                                    By6��  
�          @�G�>�
=@U@)��B{B���>�
=@]p�@�RB��B�(�                                    By6�`  �          @�Q�?\)@��H?��Ac
=B�p�?\)@�(�?Tz�A4��B���                                    By6�  �          @�Q�?!G�@l��?�Q�Aڣ�B�Ǯ?!G�@r�\?�  AÅB�=q                                    By6�  �          @���8Q�@g
=@
=qA��HB�Ǯ�8Q�@n{?�(�A�33B���                                    By6�R  
�          @�{�0��@s�
?ǮA�  BȊ=�0��@x��?�{A�z�B�#�                                    By7�  �          @���
=q@z�H?��RA�\)B��
�
=q@\)?��
A��BÏ\                                    By7�  �          @����W
=@p  ?�(�A��
B�L;W
=@vff?�\Ař�B�#�                                    By7#D  �          @�Q�>��@J=q@5B'\)B��>��@S�
@*�HB(�B�{                                    By71�  "          @��R>�G�@P  @(��BffB���>�G�@XQ�@p�B(�B�=q                                    By7@�  
�          @���>���@5�@J=qB?B���>���@?\)@@��B3p�B��)                                    By7O6  �          @�  >�(�@
=@\��B\�B��=>�(�@#33@S�
BPG�B�                                      By7]�  �          @�ff=���?��@q�B�aHB��{=���?��R@k�Bv�B�=q                                    By7l�  �          @���=u?��@l(�B|�B�ff=u@�@e�Bo\)B��q                                    By7{(  "          @��>�@5�@G�B>��B���>�@@  @=p�B2{B��                                    By7��  T          @��>k�@P��@-p�B�B��f>k�@Z=q@!G�B�RB�B�                                    By7�t  	`          @��>���@s33?�G�A�
=B�8R>���@xQ�?��A���B�k�                                    By7�  "          @�p�>.{@�  ?��A}G�B�(�>.{@��?fffAHz�B�8R                                    By7��  [          @�p�>��@�G�?��\Ab=qB�>��@��H?G�A-�B���                                    By7�f  �          @�p�>W
=@���?s33AR{B�\>W
=@�33?5AQ�B��                                    By7�  T          @�>�  @z�H?�
=A�  B��3>�  @�  ?���A�
=B��)                                    By7�  T          @��R>�z�@s�
?�G�A�z�B�ff>�z�@y��?��A�G�B���                                    By7�X  
Z          @��?
=@j�H@�A�\B��3?
=@r�\?�ffA�\)B�=q                                    By7��  �          @�\)>��R@n�R?��RA�(�B��>��R@u?�G�Aď\B���                                    By8�  T          @�ff>���@w
=?���A��B��>���@|��?�{A�G�B�Q�                                    By8J  �          @�
=>�(�@r�\?�ffA�{B��>�(�@x��?ǮA�  B�B�                                    By8*�  
�          @�  >�z�@g�@��A�{B���>�z�@p  ?�(�A�B�L�                                    By89�  "          @�  >��@hQ�@p�A�  B��>��@p��?�p�A�p�B�33                                    By8H<  T          @�{>u@dz�@
�HA�z�B�Q�>u@l��?���AݮB��{                                    By8V�  "          @��
?���@e?�{A��\B�
=?���@l(�?���A��HB��                                    By8e�  T          @���?u@hQ�?޸RA�=qB�W
?u@o\)?�  A��
B��                                    By8t.  �          @�ff?�@l��?�33A�{B�
=?�@s�
?�z�A���B��                                    By8��  �          @���>��@aG�@Q�A��B���>��@i��?�33A�  B�W
                                    By8�z  "          @�
=?z�@l(�?��HA�\)B�#�?z�@s�
?��HA���B��                                    By8�   "          @�z�>��@u�?�G�A���B��>��@z�H?�  A�p�B�G�                                    By8��  �          @�>��@mp�?���Aң�B��>��@u�?���A�=qB��H                                    By8�l  �          @�z�>�{@`  @Q�A�(�B�
=>�{@hQ�?��AۅB�u�                                    By8�  
�          @~�R=u@?\)@'�B%  B���=u@J=q@�HB�B��q                                    By8ڸ  
�          @�Q�>���@N{@Bp�B�ff>���@W�@�A�B��                                    By8�^  T          @�G�?��@Vff@
=qB�B��?��@_\)?�
=A�Q�B�z�                                    By8�  �          @�=q?333@Dz�@{B�\B�.?333@N�R@��B	{B�p�                                    By9�  �          @�=q<��
@B�\@,��B&Q�B�ff<��
@N{@�RBQ�B�p�                                    By9P  
�          @��\=��
@9��@7�B2�RB���=��
@E@*=qB"��B��)                                    By9#�  �          @�=q�\)@:�H@4z�B/�B�z�\)@G�@'
=Bz�B�#�                                    By92�  "          @}p�����@!�@@��BGp�B�� ����@/\)@4z�B7{B�(�                                    By9AB  
�          @��\��z�@{@Z=qBb�B��)��z�@p�@O\)BR\)B�k�                                    By9O�  "          @�p����?��@n�RB}z�B�\)���?�ff@g
=Boz�B�{                                    By9^�  
�          @p  ��G�?�@Tz�B��C.��G�?Q�@QG�By=qC��                                    By9m4  �          @o\)��=q>�{@W
=B�aHC%�)��=q?
=@Tz�B���C8R                                    By9{�  
�          @j=q���>�G�@]p�B�p�CJ=���?333@Z�HB�k�C��                                    By9��  �          @b�\��
==��
@Tz�B��C/�f��
=>���@S33B�33C$W
                                    By9�&  �          @g
=��(�>�=q@XQ�B��=C'�Ϳ�(�?�@VffB��fC�=                                    By9��  T          @n�R��  >�z�@_\)B�� C&���  ?\)@]p�B��RC                                    By9�r  
�          @z=q��(�?B�\@aG�B�C�׿�(�?��
@]p�Bzz�C\                                    By9�  
�          @x�ÿ�
=?z�H@W
=Bo��C�׿�
=?�p�@QG�Bf
=C�{                                    By9Ӿ  �          @}p���=q?#�
@[�Br  C Ǯ��=q?fff@W�Bk(�C��                                    By9�d  �          @|(���  ?��@7�BY{B��׿�  ?�{@.�RBJ�\B��H                                    By9�
  �          @u��8Q�@5@��B\)B�8R�8Q�@A�@{B�BΔ{                                    By9��  �          @n�R�n{@p�@(Q�B3��B�\)�n{@*=q@�B"�Bڀ                                     By:V  �          @xQ�u@'
=@,(�B0Q�B�=q�u@4z�@�RB�Bم                                    By:�  �          @qG��^�R@{@9��BJffB�녿^�R@(�@-p�B9=qB�G�                                    By:+�  �          @tz῎{@\)@*�HB1��B����{@,(�@p�B �B�Ǯ                                    By::H  T          @y����p�@'�@'�B(��B�#׿�p�@5�@��B�B��                                    By:H�  "          @��\�   ?�p�@q�B�
=B�aH�   ?��@i��B|(�B�\)                                    By:W�  T          @��;��?�G�@z�HB�33B�����?�=q@s�
B�=qB�{                                    By:f:  T          @�Q�J=q?���@W
=BmB�
=�J=q@@L��B\�B�u�                                    By:t�  �          @�Q�h��@p�@@  BC\)B�z�h��@,��@1�B1G�B��                                    By:��  �          @��׿s33@�
@\��BXffB�� �s33@&ff@P  BFffB�{                                    By:�,  "          @����O\)@
=q@g�Bf
=B�{�O\)@p�@Z�HBS�RB�p�                                    By:��  )          @����
=q@�@j�HBl�BЀ �
=q@�H@^�RBY�B�
=                                    By:�x  
S          @�=q�aG�@��@j�HBh��B��f�aG�@#�
@]p�BU\)B���                                    By:�  �          @�녾u@�@g�BdQ�B�Ǯ�u@(Q�@Y��BP��B�p�                                    By:��  
�          @����:�H@
=q@h��Bg�\B�p��:�H@{@[�BT�\B�
=                                    By:�j  �          @�z�k�@G�@n�RBj  B�� �k�@%@`��BV  B��                                    By:�  �          @���(��@�@s�
Bp�RB�ff�(��@=q@g
=B]\)B��
                                    By:��  "          @�����\?��H@���B�33B�Ǯ���\?�@xQ�Bw�B�3                                    By;\  �          @�\)���?h��@�  B��CW
���?��@���B�.C��                                    By;  �          @�
=�k�?u@���B�k�C�{�k�?��@��B�aHB���                                    By;$�  �          @���L��?���@���B�(�B�Q�L��?��H@���B�ffB�\                                    By;3N  
�          @��ÿ��?�{@���B�L�B�
=���?޸R@�z�B�u�B���                                    By;A�  �          @�  �.{?���@�
=B��B�ff�.{?�p�@��\B�{B���                                    By;P�  �          @�  ���
?��H@�(�B���C�׿��
?˅@�  B|{C �                                    By;_@  T          @��\���?h��@���B�  Cc׿��?�ff@�p�B�.CQ�                                    By;m�  �          @�
=���
>��H@��B��
C"
=���
?aG�@�33B�u�C                                      By;|�  
�          @\)����@r�\?\(�AL  B�����@vff?   @�(�B��H                                    By;�2  �          @�  �B�\@|��?�ffA�Bɮ�B�\@���?k�AJ=qB��                                    By;��  �          @��R�G�@�{?��A���B��G�@�G�?fffA;33BȀ                                     By;�~  "          @�  ��33@}p�?У�A���B�  ��33@�33?��RA�{B�Ǯ                                    By;�$  
�          @�Q쿐��@�Q�?ٙ�A��B�k�����@��?�ffA�z�B�\)                                    By;��  �          @�Q쿜(�@�  ?�Q�A��B��
��(�@�z�?��A�33BԸR                                    By;�p  "          @�\)��p�@s33?��RA�\)B���p�@~�R?���A��\B�G�                                    By;�  
�          @�Q쿴z�@l��@Q�A�B��쿴z�@x��?�  A�p�B��                                    By;�  �          @�Q쿧�@vff?�A��
Bٮ���@���?��
A��RB�.                                    By< b  �          @�Q쿫�@�Q�?�ffA���B�\���@�z�?��Aj�RB��                                    By<  
�          @�����@��\?���A���B�����@�{?uAF=qB�33                                    By<�  
�          @�����R@�33?���A�=qBճ3���R@��R?n{A?�
B��
                                    By<,T  T          @�ff���@��H?���Af�HB�����@�?.{A�B�aH                                    By<:�  
�          @�  ����@{�?�z�A���B�Ǯ����@��\?�  A�\)B�u�                                    By<I�  M          @�  ���\@y��?�A�B�Q쿂�\@�=q?�G�A�(�B��                                    By<XF  �          @��R���H@���?�G�A�Q�B�ff���H@��?Tz�A-BԔ{                                    By<f�  )          @��Ϳ�=q@��>���@��RBД{��=q@�Q�#�
��B�u�                                    By<u�  M          @�p���G�@w
=�#�
�#�
B޽q��G�@u��
=���B���                                    By<�8  
�          @�=q���H@�G�>Ǯ@�{B۳3���H@�녽#�
���HBۅ                                    By<��  T          @�G����@���?(��A�B�\)���@��\>k�@A�B���                                    By<��  T          @�(���  @�33?^�RA7�
B��f��  @�p�>�(�@���B�\)                                    By<�*  
�          @��Ϳ�33@�\)>�Q�@��B�Q쿓33@����Q쿘Q�B�8R                                    By<��  �          @�zῪ=q@�p�����Q�B�LͿ�=q@��;���  Bׅ                                    By<�v  �          @��ÿ��H@{���ff�f=qB�33���H@r�\��(����\B�aH                                    By<�  
�          @�=q�}p�@k���\��G�B�8R�}p�@\(�����RB�B�                                    By<��  
�          @�녿�  @p�׿����ȣ�B���  @b�\�{��33Bӊ=                                    By<�h  �          @����  @u���p���z�B�B���  @h�ÿ�33��\)Bҳ3                                    By=  T          @�����@u��33��
=B�ff���@j�H������{B���                                    By=�  �          @�녿k�@�{>�  @UB��H�k�@�{�aG��:=qB��H                                    By=%Z  T          @�=q�=p�@�\)>�33@��B���=p�@������
=B�Ǯ                                    By=4   "          @�zΌ��@��H?�RA�HB������@�z�>#�
@�BڸR                                    By=B�  
�          @�{���R@���?�\@ӅBۣ׿��R@�=#�
>�B�\)                                    By=QL  
�          @�p���  @��
?
=q@�\B��ÿ�  @��=�\)?k�Bۨ�                                    By=_�  T          @�{���\@�z�?#�
A�B�.���\@�{>#�
@�B��
                                    By=n�  �          @�\)��ff@{���p���  B�(���ff@n�R����
=B�8R                                    By=}>  �          @�ff��z�@W
=�z���B� ��z�@Dz��,(��{B�Q�                                    By=��  T          @�p���ff@w
=�У�����B�(���ff@h���z���p�B�33                                    By=��  
�          @��ÿ��@vff��=q����B٨����@j�H��\�îB�ff                                    By=�0            @��׿�\)@o\)��  ��33B����\)@b�\��
=���B�33                                    By=��  �          @��׿��R@r�\��(����B�  ���R@g���z����B��
                                    By=�|  �          @�Q�Ǯ@s�
����h  B�ff�Ǯ@j=q��G����\B�{                                    By=�"  
�          @���\@r�\�����w�B��)�\@g���=q����B��                                    By=��  �          @�����@g������p(�B�����@^{�\����B��                                    By=�n  �          @�Q��ff@o\)�k��H��B� ��ff@fff��\)��ffB�=q                                    By>  �          @�\)��(�@mp�����h��B�׿�(�@c33��G����B�\                                    By>�  �          @�p���p�@a녿��\�bffB�\��p�@XQ쿹����{B�R                                    By>`  [          @��\?�\)@s33?5A��B�?�\)@w
=>u@L��B�=q                                    By>-  M          @���?�@~�R?�@�B���?�@���=�\)?z�HB�                                    By>;�  
�          @���>�p�@����p�����B��>�p�@���c�
�@z�B��q                                    By>JR  �          @��\?\(�@��R=u?G�B��H?\(�@�{����ȣ�B��q                                    By>X�  "          @�33?5@���>.{@�\B���?5@�  ��p����HB��                                    By>g�  
�          @���?8Q�@�
=���
�uB��=?8Q�@�������B�\)                                    By>vD  T          @��?G�@�  >��
@���B�=q?G�@�Q�aG��:=qB�G�                                    By>��  
�          @�33?8Q�@�  ?��@�=qB���?8Q�@���<#�
=��
B���                                    By>��  
�          @�=q>�{@��?:�HA�B���>�{@���>8Q�@
=B��                                    By>�6  
�          @�33>���@���?!G�A�B�33>���@�=q=�\)?n{B�Q�                                    By>��  �          @�33>��@�\)?aG�A<(�B��)>��@��>���@��B���                                    By>��  	�          @��=�Q�@���?��Adz�B�k�=�Q�@�Q�?�\@���B�z�                                    By>�(  
�          @�G�>���@{�?�z�A��RB�G�>���@��?���Aup�B���                                    By>��            @��?�R@w�?�p�A�{B��R?�R@�=q?���A�(�B��{                                    By>�t  
Z          @��?&ff@{�?�33A��
B�(�?&ff@��?�{Ao\)B���                                    By>�  
�          @��\?+�@~{?�=qA��B��R?+�@�z�?��A^=qB��                                    By?�  
�          @��\?G�@}p�?ǮA���B��q?G�@�z�?��\AX��B���                                    By?f  T          @��?Y��@}p�?���A��B���?Y��@��
?fffAA�B��{                                    By?&  �          @���?Q�@���?�Q�A��B�  ?Q�@���?#�
A	�B��                                    By?4�  
Z          @����=q@.�R@5�B'B����=q@E@�BG�B�
=                                    By?CX  
�          @�\)��  @*�H@:=qB.{B����  @C33@!G�BG�B�z�                                    By?Q�  [          @�
=��{@*=q@>�RB3�RB�#׿�{@B�\@%�B=qB��                                    By?`�            @�녿��
@,��@FffB7�B��
���
@Fff@,(�B�B���                                    By?oJ  
�          @��ÿ���@&ff@FffB:  B��ÿ���@@  @-p�BffB�\)                                    By?}�  
�          @�Q쿨��@0��@8��B-z�B��Ῠ��@H��@�RB=qB�=                                    By?��  
�          @��׿0��@S�
@&ffBQ�B�k��0��@h��@
=A�p�B�W
                                    By?�<  T          @��R�.{@U�@�RBp�B�G��.{@h��?�p�A�p�B�L�                                    By?��  [          @��R�fff@J�H@'�B=qBӣ׿fff@`  @	��A��BиR                                    By?��  �          @�\)�:�H@\��@ffB��B��ÿ:�H@o\)?�=qA�G�B��                                    By?�.  	�          @�z��@^{@��B ��B�  ��@o\)?�Q�A�=qBýq                                    By?��  
�          @��Ϳ��@333@3�
B+B�ff���@J�H@Q�B�Bۙ�                                    By?�z  
�          @�z῜(�@>{@'�B��B�׿�(�@S�
@
=qA��
B�aH                                    By?�   
Z          @�z῁G�@HQ�@ ��Bp�Bי���G�@]p�@�A��B�ff                                    By@�  
�          @�z�@  @W
=@�B�HB�Q�@  @j=q?�G�A�Q�B�W
                                    By@l  
Z          @�zᾞ�R@p��?�{A�
=B�\)���R@|��?��Al  B��f                                    By@  T          @��c�
@#�
@J�HBF
=Bڀ �c�
@?\)@0��B%�HB�33                                    By@-�  	�          @�
=�Y��@ff@Y��BV�B���Y��@4z�@@��B6p�Bը�                                    By@<^  
�          @�  �L��@G�@i��Bl��B�ff�L��@"�\@S�
BLz�B�#�                                    By@K  T          @�Q�Q�?�(�@j�HBoG�B�{�Q�@   @UBO�B�L�                                    By@Y�  	�          @�
=��33?�Q�@c33BfQ�B����33@��@N{BG��B�#�                                    By@hP  
�          @�p����R@�@L(�BG�B�����R@5�@333B(�RB�Q�                                    By@v�  �          @���}p�@�R@W�BX�B��f�}p�@-p�@?\)B8G�B��                                    By@��  "          @�(��Tz�@{@L(�BK
=B�33�Tz�@;�@1�B)�BӸR                                    By@�B  T          @��
�\)@1G�@@  B:�B��\)@L��@#33Bz�B�                                    By@��  �          @��Ϳ�@8Q�@:�HB3ffB�=q��@S33@��B�RBǀ                                     By@��  "          @�(��
=q@=p�@5�B,��BȀ �
=q@W
=@B	�B��                                    By@�4  �          @��
��R@@��@/\)B&�RB�uÿ�R@X��@\)B��B��
                                    By@��  "          @�33�&ff@Fff@%B33B˳3�&ff@^{@�A�Q�B�B�                                    By@݀  
�          @�33�+�@<��@0  B(��B͸R�+�@Vff@��B�HB���                                    By@�&  
�          @��H�5@(Q�@A�B@�B�aH�5@E�@%�B(�B�(�                                    By@��  T          @�녿��@8��@(Q�B"33B�ff���@QG�@	��A��B�
=                                    ByA	r  �          @��
��33@(Q�@<(�B7
=B�B���33@C�
@\)BQ�B�33                                    ByA  �          @�zῺ�H@�H@@  B:�\B�8R���H@7�@%�B��B�                                    ByA&�  �          @�\)���@��@N�RBGB����@7�@3�
B&�
B�Q�                                    ByA5d  
'          @�G���p�@��@XQ�BO�B�  ��p�@-p�@?\)B0  B�\)                                    ByAD
  "          @��ÿ�33@��@Q�BGB��q��33@,��@8Q�B){B��                                    ByAR�  
�          @��׿��
@	��@N{BD(�C�����
@(��@5�B&Q�B���                                    ByAaV  
�          @��H�У�@%�@E�B5  B�Ǯ�У�@B�\@'�B�RB�                                     ByAo�  
�          @��H���H@2�\@5B#��B�{���H@N{@ffB
=B�{                                    ByA~�  
�          @��ÿ�
=@333@1G�B!Q�B��)��
=@N{@�B p�B��                                    ByA�H  "          @�  ��Q�@7
=@p�B�
B�k���Q�@N{?��HA��B�(�                                    ByA��  
(          @�\)��  @<(�@�RB�B񞸿�  @S�
?��HA݅B���                                    ByA��  T          @�\)���R@[�@��A�B۳3���R@n�R?��A�p�Bب�                                    ByA�:  
(          @������@Fff@'�Bp�B�q���@`  @�
A癚B�\                                    ByA��  �          @�녿�ff@=p�@.{Bp�B�aH��ff@W�@�A�B�u�                                    ByAֆ  "          @�Q쿺�H@W�@��A�Q�B����H@k�?�ffA���B�ff                                    ByA�,  T          @���  @N�R@
�HA�
=B��H��  @c33?���A�{B��H                                    ByA��  "          @�p���p�@HQ�@�B��B�R��p�@^�R?�G�A�(�B�33                                    ByBx  T          @�ff��(�@E@�B{B�\)��(�@[�?�(�A���B�aH                                    ByB  "          @��Ϳ���@:�H@�
B�B��H����@QG�?��
A�G�B�.                                    ByB�  "          @��ͿУ�@<��@=qBG�B���У�@Tz�?�\)A֏\B�k�                                    ByB.j  "          @�ff���@'�@"�\B��B�����@AG�@33A�Q�B�W
                                    ByB=  T          @��
�\(�>�z�@)��B
=C/=q�\(�?=p�@#�
B33C'�)                                    ByBK�  �          @y���!G�?�ff@!�B'
=C�H�!G�?�(�@G�Bp�C��                                    ByBZ\  �          @��?�@e?8Q�A-��B��f?�@j=q=�Q�?�ffB��=                                    ByBi  T          @��R@��@dz�>�Q�@���BiQ�@��@e���=q�k�Bi�\                                    ByBw�  T          @��\@
�H@n{>�ff@���Bo{@
�H@o\)�aG��6ffBo��                                    ByB�N  �          @��@�@k�>��@ȣ�Bmz�@�@mp��B�\��RBn(�                                    ByB��  �          @�  ?�
=@n{?8Q�AQ�Bz33?�
=@q�=L��?333B{�                                    ByB��  "          @��R?��@l(�?:�HA
=B|{?��@p��=u?L��B}��                                    ByB�@  �          @��?�ff@q�?+�AG�B�{?�ff@u���
�W
=B�                                    ByB��  �          @�Q�?���@w�?�  A�33B�z�?���@���?
=q@���B�Ǯ                                    ByBό  
�          @�G�@�@g�?�ffAb=qBr��@�@p  >\@��Bv33                                    ByB�2  �          @���?��R@hQ�?n{AK\)Bu=q?��R@o\)>��@c�
Bx{                                    ByB��  �          @���?�Q�@j=q?s33AN�\BxQ�?�Q�@qG�>�=q@i��B{(�                                    ByB�~  �          @�=q?�  @o\)?�33Ax��B�?�  @x��>�ff@���B�p�                                    ByC
$  �          @���@�\@e�?��An{Bq\)@�\@n{>�
=@�{Bu{                                    ByC�  
�          @�\)?�(�@^{?���A�=qB~G�?�(�@k�?J=qA0Q�B��q                                    ByC'p  
�          @���?��H@]p�@�A�\)B�W
?��H@qG�?�{A���B���                                    ByC6  "          @���?�33@e?�{A�Q�B�G�?�33@w�?�A}�B�#�                                    ByCD�  T          @��?�z�@mp�?�Q�A��RB�=q?�z�@}p�?xQ�AP��B���                                    ByCSb  �          @��?�33@u�?��A���B��f?�33@���?(��AQ�B��                                    ByCb  
�          @���?��@s33?�z�A�  B���?��@�  ?.{A��B���                                    ByCp�  \          @���?��H@q�?�
=A�p�B���?��H@~�R?333A��B���                                    ByCT  
Z          @�=q?�@}p�?��RA��
B�.?�@�(�>��@ə�B�ff                                    ByC��  
�          @�G�?ٙ�@k�?���A���B�aH?ٙ�@w�?(��A�\B���                                    ByC��  *          @���?�@e?�
=A��B{�R?�@s33?8Q�A33B�ff                                    ByC�F  �          @��
?��@�Q�?G�A$��B��?��@��\    =L��B���                                    ByC��  
�          @���?�Q�@|��?n{AB�\B��q?�Q�@��>��@ ��B���                                    ByCȒ  \          @�=q?�@tz�?G�A'�B�u�?�@y��=#�
?   B�aH                                    ByC�8  �          @�=q?���@w
=?�RA(�B���?���@y������G�B�\)                                    ByC��  
�          @�=q?�\@y��?
=@��B�8R?�\@|(��8Q��ffB���                                    ByC�  �          @�=q?��@���?0��A  B��f?��@��\���Ϳ�=qB�ff                                    ByD*  �          @��?\@}p�?E�A$(�B��f?\@��ü��
�aG�B���                                    ByD�  T          @��H?�G�@~�R?Tz�A/�
B�W
?�G�@��<�>�G�B�#�                                    ByD v  �          @�=q?��@�=q?�@��HB�L�?��@���u�J�HB��\                                    ByD/  �          @��H?��@��\?(�AG�B��{?��@��
�L���-p�B��                                    ByD=�  
(          @��H?��H@���?(��Az�B�(�?��H@��\�����
=B���                                    ByDLh  
�          @�Q�?�G�@vff>���@��
B���?�G�@vff��p�����B���                                    ByD[  "          @�33?fff@g
=@G�A��\B��?fff@\)?�  A�G�B���                                    ByDi�  "          @��\?G�@e�@33B z�B�u�?G�@~{?��A���B�Ǯ                                    ByDxZ  "          @��\?J=q@j=q@�A�B��?J=q@���?�33A�33B���                                    ByD�   �          @��H?޸R@p  ?�ffA�  B�#�?޸R@|(�?�@�{B�8R                                    ByD��  "          @��?��@o\)?�\)A�z�B��q?��@|(�?
=@��B��f                                    ByD�L  
�          @��H?�Q�@qG�?h��AA�B{G�?�Q�@xQ�>�?ٙ�B}�
                                    ByD��  "          @���?��@qG�?:�HA{B~33?��@vff�L�Ϳ#�
B�H                                    ByD��  �          @�=q?��H@z=q?�{Am�B�(�?��H@���>�z�@u�B��                                     ByD�>  T          @��H?�z�@{�?�Q�A}�B���?�z�@��H>�Q�@��B�
=                                    ByD��  	�          @�?��@��?xQ�AI�B���?��@�
==���?���B��3                                    ByD�  	�          @�?�33@�33?c�
A:=qB�\)?�33@�{<�>\B�(�                                    ByD�0  \          @�33?��\@��\?O\)A,  B�\)?��\@���L�Ϳ#�
B���                                    ByE
�  L          @��?���@~{?z�HAS�B�Q�?���@��H>\)?���B�L�                                    ByE|  
�          @�z�?�Q�@|(�?��A�=qB���?�Q�@��
>�G�@�\)B�p�                                    ByE("  T          @��
?�=q@xQ�?�  A��B�Ǯ?�=q@���>��@�ffB��                                    ByE6�  T          @�33?�Q�@z�H?�Q�A}B��{?�Q�@��\>�{@��B�\                                    ByEEn  
Z          @�33?���@��\?W
=A333B�=q?���@�p�����ffB��H                                    ByET  �          @�z�?��H@���?��A[�B��?��H@���>.{@G�B�=q                                    ByEb�  �          @��\?�p�@�G�?fffAAG�B�(�?�p�@�z�<�>ǮB��                                    ByEq`  
�          @��?�  @���?�G�AV�RB���?�  @�p�>�?��HB��{                                    ByE�  �          @��?�\)@{�?�p�A�  B�z�?�\)@�33>�p�@��
B���                                    ByE��  "          @�z�?u@�=q?��A�
=B�� ?u@�  >Ǯ@��
B��{                                    ByE�R  �          @�(�?h��@���?�A�=qB��{?h��@��?�@޸RB���                                    ByE��  �          @�33?���@���?�(�A���B��\?���@�ff>���@��B��q                                    ByE��  
�          @�(�?��@���?�=qAd��B��{?��@�>B�\@�HB��                                    ByE�D  �          @��
?�=q@~�R?���A��
B��=?�=q@�{>��H@��B���                                    ByE��  �          @�z�?�@~�R?�z�A�=qB�Q�?�@�{?�\@�\)B��H                                    ByE�  T          @�{?�ff@�=q?�z�A��RB�#�?�ff@���>��H@���B��                                     ByE�6  
�          @�
=?�p�@��\?���A�G�B�\)?�p�@���>Ǯ@�=qB�Ǯ                                    ByF�  "          @�
=?�=q@�?�33Ao33B�\?�=q@��\>W
=@,(�B�
=                                    ByF�  �          @�Q�?��@�{?��
A��B�� ?��@��
>��
@�(�B���                                    ByF!(  	�          @��?xQ�@��\?�ffAT��B���?xQ�@�ff=�\)?^�RB�L�                                    ByF/�  
�          @���?k�@�=q?��\APQ�B�?k�@�=#�
?�B���                                    ByF>t  �          @��?�G�@��?��AZffB�=q?�G�@��
=�Q�?�
=B�                                      ByFM  "          @���?k�@���?��AT(�B��?k�@�=L��?.{B��{                                    ByF[�  �          @���?k�@�33?uAB=qB�?k�@��R�#�
���B��\                                    ByFjf  T          @���?z�H@��\?c�
A5�B�p�?z�H@�p����Ϳ�ffB��                                    ByFy  �          @�\)?��@���?=p�A�B�W
?��@��H�u�EB��3                                    ByF��  
�          @�z�?fff@�  ?�RAffB�33?fff@��þ�33���\B�\)                                    ByF�X  
�          @�33?Tz�@�\)?   @ҏ\B��H?Tz�@�\)�����\)B��f                                    ByF��  T          @���?h��@���\)�aG�B�u�?h��@�녿�G��Z�RB��q                                    ByF��  �          @��?��\@�33�.{��\B��=?��\@|�Ϳ����l��B�W
                                    ByF�J  �          @��?(�@��H>�@�ffB�  ?(�@��\�
=q���HB���                                    ByF��  �          @�z�?��@��?
=@�G�B��\?��@��\�����=qB���                                    ByFߖ  "          @�(�?333@�Q�?!G�Az�B�\)?333@�G���p�����B��                                     ByF�<  T          @�=q?n{@�{>�
=@�(�B�{?n{@������33B���                                    ByF��  �          @���>Ǯ@�Q�>��R@��B���>Ǯ@�
=�.{��RB��                                    ByG�  �          @��\    @��׿z���{B��    @��׿�ff��z�B��                                    ByG.  T          @��H=L��@��ÿ.{��B���=L��@�Q��33��
=B��\                                    ByG(�  �          @��>�  @�  ����(�B�.>�  @�  �����z�B�                                    ByG7z  �          @�  >\)@�ff�����p�B�G�>\)@}p���G���33B�\                                    ByGF   �          @�Q��@��R�&ff�\)B�ff��@|(���{���
B���                                    ByGT�  �          @�\)����@��
�^�R�>=qB�8R����@s33����G�B��                                    ByGcl  �          @�
=��ff@�33�^�R�>�\B����ff@q녿���p�B��\                                    ByGr  �          @��þ�=q@�{�#�
�\)B�p���=q@�녿��
�b�HB���                                    ByG��  
�          @�33?
=@�
=?Tz�A0Q�B���?
=@�G��B�\�!�B�B�                                    ByG�^  
(          @��>���@��R?=p�A (�B��>���@�Q쾊=q�j�HB�8R                                    ByG�  
�          @���=�G�@�Q�>�@ÅB���=�G�@�  ����z�B��                                    ByG��  T          @�Q�>.{@�\)>\@��B�u�>.{@�ff�#�
�
=B�p�                                    ByG�P  �          @�ff��@�>�33@�Q�B��3��@�z�(���G�B��3                                    ByG��  T          @�zᾏ\)@�33>�
=@��B�녾�\)@��H�����B��                                    ByG؜  T          @�zᾣ�
@��\?��@��HB�𤾣�
@�33��ff��G�B��                                    ByG�B  �          @����z�@��\?B�\A'\)B�33��z�@�zᾀ  �^{B�\                                    ByG��  �          @��    @���?��
Ad��B�      @���<��
>��RB�                                      ByH�  
Z          @�p��#�
@�Q�?��A~�RB�G��#�
@��>\)?��B�8R                                    ByH4  
�          @�{����@���?���AuG�B������@�p�=�Q�?�G�B��R                                    ByH!�  �          @���?+�@�{>��H@У�B���?+�@������p�B�Ǯ                                    ByH0�  "          @�Q�?��@��>W
=@7
=B�\?��@�G��L���-p�B��\                                    ByH?&  T          @��?���@~�R��Q쿜(�B��)?���@u�����hQ�B��                                     ByHM�  �          @�Q�?˅@|�;���z�B�(�?˅@r�\��\)�s�
B��                                     ByH\r  T          @��?��\@�33>��?�(�B��?��\@�Q�^�R�=�B�z�                                    ByHk  T          @��?\(�@�z�>L��@1G�B���?\(�@�녿Q��3�B�33                                    ByHy�  �          @��?��@�33>B�\@#33B���?��@��׿Tz��5B��                                    ByH�d  �          @���?�\)@��
>���@��\B��?�\)@�=q�:�H���B�L�                                    ByH�
  �          @��R?���@y��?8Q�A{B���?���@}p���\)�s33B��                                    ByH��  T          @�?�=q@z�H>��H@�ffB�p�?�=q@z�H��\�߮B�k�                                    ByH�V  T          @��?��@g����H�ڏ\BzQ�?��@XQ쿰����
=Bs�
                                    ByH��  �          @��\@!�@Y���W
=�3\)BU�@!�@E�������BJ�R                                    ByHѢ  �          @��@�@`�׿
=q��B](�@�@P�׿�z�����BU(�                                    ByH�H  
Z          @���@{@hQ�
=q�陚Bj=q@{@W���Q���G�Bb�R                                    ByH��  
(          @�=q@�@hQ�Q��/\)Bl{@�@S33���H���Bb\)                                    ByH��  T          @���@�@l(������HBp�@�@[���p����
Bi                                      ByI:  T          @���@�
@n{��(����Bt  @�
@_\)��\)���HBm��                                    ByI�  �          @���?�(�@o\)�����(�Bx?�(�@^�R��p�����Bq�R                                    ByI)�  "          @���@z�@l(������Br�@z�@Z�H���R���
Bkp�                                    ByI8,  
�          @���@	��@j=q���ƸRBnff@	��@Z�H�����\)Bgz�                                    ByIF�  �          @�Q�?���@o\)�   �أ�By��?���@_\)��������Bs{                                    ByIUx  \          @���?�
=@vff�.{�
=B���?�
=@c33��33��Q�B�k�                                    ById  "          @���?�{@x�ÿ8Q��\)B�\?�{@dz�ٙ�����B��                                     ByIr�  
�          @��@!�@XQ�L���.�RBU  @!�@Mp������jffBOff                                    ByI�j  \          @���@   @_\)�B�\� ��BY��@   @U�����jffBTff                                    ByI�  L          @�G�@"�\@[����R��(�BU�
@"�\@N�R��Q���
=BOG�                                    ByI��  �          @�G�@%@Z=q�u�N{BS{@%@N�R��\)�rffBM
=                                    ByI�\  *          @���@p�@fff�333�ffBj  @p�@S33��\)��\)B`�R                                    ByI�  �          @�G�@�@k��:�H���Btz�@�@W
=��z�����Bk�                                    ByIʨ  T          @�=q?�\)@u��R�Q�B��?�\)@b�\��{����Bxz�                                    ByI�N  
b          @�33@p�@mp���ff���Bl�H@p�@]p���z���G�Be                                    ByI��  ~          @��\@
�H@mp�����33Bn��@
�H@\(���p����Bf��                                    ByI��  �          @��\@
�H@k��5�{Bm��@
�H@W
=��33����Bd�R                                    ByJ@  �          @��H@
�H@j=q�W
=�1�Bm\)@
�H@S�
��\��p�Bb��                                    ByJ�  T          @��@�\@l�;aG��7
=Bh�H@�\@`  ��Q��}�Bcff                                    ByJ"�  �          @��@  @j�H�8Q��
=Bi�@  @_\)��33�w
=Bd                                    ByJ12  "          @��\@G�@k��aG��>{Biz�@G�@_\)������z�Bc�H                                    ByJ?�  T          @�=q@p�@j=q�����\BkG�@p�@XQ��G����
Bc{                                    ByJN~  
�          @�=q@p�@j�H�������Bk�H@p�@[���\)���\Bd�H                                    ByJ]$  
�          @���@z�@l�Ϳ   ��
=Bsff@z�@[���p���Bk�
                                    ByJk�  "          @�G�@(�@Y���aG��=G�BY�@(�@B�\��G���ffBM�                                    ByJzp  
(          @�G�@�
@\�Ϳz�H�T��B`@�
@C�
����иRBS�                                    ByJ�  T          @���?�{@u�fff�C�
B��?�{@\�Ϳ�z�����B�                                    ByJ��  T          @���?�@k�����j=qBy�R?�@P  ��\��G�Bm��                                    ByJ�b  T          @�Q�?�@g
=��
=��=qBxG�?�@J=q�
=��{Bk
=                                    ByJ�  �          @��?���@c33��  ��\)Bt�R?���@Dz��
�H��Q�BfG�                                    ByJî  
Z          @�p�?��@dzῆff�h��Bx(�?��@H�ÿ�p���RBk�
                                    ByJ�T  
�          @�
=?��@dz῝p���p�By  ?��@Fff�
=q��{Bk�                                    ByJ��  
�          @�Q�@
=@aG���z��|z�Bl  @
=@C�
����p�B]��                                    ByJ�  
�          @�Q�@Q�@_\)��
=��(�Bjz�@Q�@B�\����B[                                    ByJ�F  
           @�G�@z�@g
=�xQ��R=qBp�@z�@Mp���
=��z�Bd�                                    ByK�  �          @�33?�p�@u����33Bz��?�p�@dz��  ���
Bs�                                    ByK�  	�          @�=q@�@q녾�(���Bw
=@�@`�׿�����Bo��                                    ByK*8  �          @�G�?��H@p  ��R�(�By�?��H@\(��У����Bq                                      ByK8�  T          @���?�@u��z��xQ�B��H?�@g
=�������B|�                                    ByKG�  T          @�Q�?���@|(�>��
@��HB��{?���@w��J=q�*�\B��                                    ByKV*  T          @���?��@y��>B�\@�RB��q?��@s33�h���C\)B��\                                    ByKd�  "          @��@ ��@s33������Bx��@ ��@h�ÿ�=q�j{Bt�\                                    ByKsv  �          @�Q�?�  @w�<�>�G�B�aH?�  @n�R��ff�ep�B��R                                    ByK�  "          @�  ?�\)@q�=�Q�?�Q�B~�H?�\)@i���z�H�V=qB{�R                                    ByK��  �          @�ff?��@u�>�{@�{B��R?��@qG��@  �$��B�{                                    ByK�h  
�          @�{?�@e=���?�33Bwp�?�@^{�k��O�
Bt=q                                    ByK�  
�          @�\)@  @^{�G��)Bd\)@  @G
=��(���\)BX�                                    ByK��  
Z          @�Q�@�R@dz�#�
�
=qBg@�R@O\)��\)���B]�                                    ByK�Z  �          @�\)@33@fff��p���33Bq�@33@Vff��\)��ffBj=q                                    ByK�   \          @�{?�ff@qG�=#�
?(�B�\?�ff@hQ쿃�
�c�
B~�                                    ByK�  
�          @�?�@n{���
��33Bz�?�@b�\����|Q�Bv
=                                    ByK�L  T          @�{?�
=@s�
<�>��B�k�?�
=@j=q��ff�jffB��R                                    ByL�  "          @�\)?���@n{����Q�Byp�?���@\(���G�����Bq�R                                    ByL�  T          @��R?�{@~�R>8Q�@ ��B�W
?�{@w
=�u�S\)B�Q�                                    ByL#>  
�          @��?}p�@�33>��@���B��q?}p�@����G��*�\B�aH                                    ByL1�  
Z          @���?���@}p�?}p�AV�HB�33?���@��\�B�\�$z�B��                                    ByL@�  �          @�G�?�{@���?��AG�B���?�{@��׿z�����B���                                    ByLO0  �          @�G�?��@��>��R@��
B�k�?��@~�R�aG��<z�B��3                                    ByL]�  "          @���?�p�@���?333A�\B�{?�p�@������H���B�Q�                                    ByLl|  �          @�Q�?�=q@~{?��AnffB�u�?�=q@�����Ϳ���B��                                     ByL{"  \          @�\)?��\@y��?�  A�\)B���?��\@�33=u?W
=B�                                      ByL��  �          @�  ?p��@z=q?���A�B��
?p��@�z�>B�\@!G�B�Q�                                    ByL�n  T          @�  ?k�@u?�ffA�33B�.?k�@��
>�Q�@���B���                                    ByL�  �          @��?u@u�?��A�(�B���?u@��>�33@���B��
                                    ByL��  
�          @��?p��@~{?�{As�B�p�?p��@��
���Ϳ��B�aH                                    ByL�`  �          @���?J=q@�G�?���AiG�B��q?J=q@�p��#�
�Q�B�p�                                    ByL�  "          @�  ?Tz�@|(�?�{A�p�B�\?Tz�@��>��?�p�B�Q�                                    ByL�  �          @�Q�?.{@{�?�  A��\B�W
?.{@�ff>�\)@n{B��\                                    ByL�R  �          @�  ?z�@w
=?�z�A�=qB��?z�@�>�ff@��HB�8R                                    ByL��  �          @�  ?\)@s33?�A��B�Q�?\)@�p�?(�A(�B�                                    ByM�  �          @�p�?=p�@fff?���A���B���?=p�@���?J=qA0(�B���                                    ByMD  
�          @��
?J=q@^{@33AB�k�?J=q@z�H?n{AQ�B�G�                                    ByM*�  	�          @��\?��H@h��?��
Al(�B�=q?��H@q녽�Q쿦ffB�                                    ByM9�  T          @��?�ff@b�\?��
Ah  B|�?�ff@l(���\)�z�HB��                                    ByMH6  
�          @���?333@e?�z�A�\)B�{?333@z�H?�@���B��f                                    ByMV�  
�          @�Q�?8Q�@k�?�A��B���?8Q�@|(�>��@i��B�aH                                    ByMe�  "          @�=q?G�@p  ?�\)A�33B�z�?G�@~�R>B�\@(Q�B��
                                    ByMt(  "          @��?O\)@r�\?�z�A�
=B��H?O\)@~{������B��f                                    ByM��  
�          @�=q?��@q�?�G�Ae�B�p�?��@y���.{�ffB�z�                                    ByM�t  	�          @�33?�p�@r�\?fffAL  B�?�p�@x�þ�\)�x��B��
                                    ByM�  �          @�p�?��R@s33?��A~�RB��?��R@~{�L�ͿB�\B�aH                                    ByM��  
�          @�{?��@q�?�33A�{B�z�?��@���>L��@,(�B�k�                                    ByM�f  ~          @�p�?Q�@tz�?���A�
=B���?Q�@��\>k�@L(�B�(�                                    ByM�  �          @�(�>u@j�H?���AծB�aH>u@��?(��A�\B��                                    ByMڲ  �          @�zᾳ33@J=q@#�
B�B�����33@r�\?�p�A�{B��=                                    ByM�X  "          @�{�c�
@7
=@:�HB1(�B�p��c�
@g
=?�z�A�=qBϔ{                                    ByM��  
�          @�ff�
=@<(�@;�B1p�BʸR�
=@l(�?�33A׮B�.                                    ByN�  T          @����@B�\@6ffB+�RB�B����@p��?��A���B�\)                                    ByNJ  
�          @�{?J=q@z�H?�  A�
=B�#�?J=q@��
    ���
B�33                                    ByN#�  "          @���?�R@s33?�=qA�z�B�W
?�R@���=�G�?\B�\)                                    ByN2�  �          @�33=u@p��?���A��HB�G�=u@��>�{@�=qB�k�                                    ByNA<  T          @|�ͼ��
@`��?�  A�{B�� ���
@w�?
=A
=B�u�                                    ByNO�  �          @�Q�>aG�@mp�?�  A���B�#�>aG�@\)>�z�@�(�B���                                    ByN^�  �          @���>u@n�R?ǮA�33B���>u@���>�{@���B�.                                    ByNm.  �          @���>��R@k�?���A��B�L�>��R@\)>���@�z�B�\                                    ByN{�  
�          @~{>��R@g�?���A�p�B�8R>��R@{�>\@���B�                                      ByN�z  �          @w
=>�\)@\(�?�Q�A���B��3>�\)@s33?��A
=B��\                                    ByN�   
�          @n�R?0��@W�?�
=A��B��)?0��@i��>��R@��\B��\                                    ByN��  
�          @h��?�\@Dz�?���A��B�Q�?�\@^�R?G�AHQ�B��                                     ByN�l  
�          @j�H>�z�@C�
?��RB  B�.>�z�@aG�?s33ApQ�B���                                    ByN�  "          @b�\>�{@7�@�BQ�B�L�>�{@W
=?��A��HB�G�                                    ByNӸ  �          @hQ�=#�
@333@�\B{B�B�=#�
@W�?��A�G�B��\                                    ByN�^  T          @fff>�Q�@B�\?��A�(�B�u�>�Q�@^�R?W
=AW�
B�#�                                    ByN�  
�          @h�þ�Q�@{@5BNz�B�33��Q�@?\)@ ��Bz�B��\                                    ByN��  
�          @n�R�W
=@L(�?�p�AݮBр �W
=@dz�?&ffA ��B�u�                                    ByOP  �          @u�+�@N{?�(�A��HB�\)�+�@k�?\(�AO�BȊ=                                    ByO�  ~          @z�H�!G�@N{@Q�B�HB�#׿!G�@n�R?�G�Ap(�B�33                                    ByO+�  �          @p  �+�@,��@�HB${B��+�@U�?���A�p�B��)                                    ByO:B  �          @i���!G�@?\)?���A�B�Ǯ�!G�@Z=q?J=qAL(�B��                                    ByOH�  T          @a녿�=q@1�?��A��B�8R��=q@I��?0��A5p�B�Ǯ                                    ByOW�  \          @i����@B�\?�(�A���B��῵@Vff>�(�@ٙ�B��f                                    ByOf4            @j�H��z�@,(�?�  A��B�p���z�@Fff?O\)AN�RB�\)                                    ByOt�  �          @`  ��\@  ?�(�A��C5���\@*�H?k�AuB���                                    ByO��  
Z          @]p��{@!�?   A  CE�{@#�
���
��{C�                                    ByO�&  	�          @\(��&ff@   ?��A���C��&ff@�R>�p�@�Cp�                                    ByO��  "          @S�
�;�?�  ���\)Cٚ�;�?�z��\�C8R                                    ByO�r  
�          @R�\�1G�?�33?#�
A4(�C8R�1G�?�\=��
?�(�Cs3                                    ByO�  
�          @X���.{?�p�?�=qA�p�CǮ�.{?�?�{A���C�\                                    ByO̾  T          @_\)�?\)?
=?ٙ�A��C(���?\)?��?�A���C�)                                    ByO�d  �          @^�R�6ff?���?�Q�A�{CB��6ff?˅?}p�A�ffC�\                                    ByO�
  �          @`����?ٙ�?�Bz�Cs3��@{?��
A��
C��                                    ByO��  �          @]p���@z�?�p�A��Cs3��@ ��?xQ�A���C��                                    ByPV  �          @]p��G�@   ?5A?
=CO\�G�@%����\C8R                                    ByP�  �          @W���(�@,��>�Q�@�
=B�Q��(�@*�H���=qB��
                                    ByP$�  
Z          @W���Q�@)��>��A33B�LͿ�Q�@*=q������\)B�\                                    ByP3H  "          @^{�.�R?�
=?fffAp  C�R�.�R@
=>k�@q�C5�                                    ByPA�  T          @g
=�8Q�@�\?@  A?�C���8Q�@
�H=L��?Q�C��                                    ByPP�  
�          @h���G
=?˅?xQ�Ax(�C�
�G
=?���>���@���C��                                    ByP_:  
(          @i���G�?��
?�ffA�z�C!�R�G�?�p�?��A�  C��                                    ByPm�  T          @p���P��?O\)?���A�Q�C&��P��?��\?�(�A�=qC��                                    ByP|�  �          @l�����@Q�?�33A�
=Cٚ���@,��?\)A�
C+�                                    ByP�,  "          @o\)�W
=@333@��B
=B�LͿW
=@XQ�?�p�A�(�B���                                    ByP��  
�          @l�Ϳ���@9��?�(�B(�Bܔ{����@XQ�?n{Aj{B�.                                    ByP�x  
�          @o\)�  @Q�?��A�(�Ch��  @0  ?0��A/�CE                                    ByP�  �          @p  ��@  ?�ffA���C	&f��@(��?=p�A9p�C��                                    ByP��  "          @q녿�@E?У�A��B�G���@\��?
=qA33B���                                    ByP�j  T          @q녾�@aG�?��\A�=qB��f��@j=q�8Q��5�B�W
                                    ByP�  �          @l�Ϳ�{@N�R?�33A�G�B���{@Z�H<�?�B�L�                                    ByP�  �          @i���G�@$z�?�  A�Q�C�=�G�@5�>�{@��C �q                                    ByQ \  
�          @p  �{@$z�?�
=A�ffCٚ�{@4z�>�=q@�CB�                                    ByQ  �          @o\)�<��?�=q?���A�\)C!H�<��@�?333A-��Cz�                                    ByQ�  "          @c33�*�H?��?�p�Ař�C&f�*�H@
�H?O\)AS�
C��                                    ByQ,N  T          @a��7�?0��?�33B  C&n�7�?��
?ǮA�  C�                                    ByQ:�  "          @e��1G�?Q�?�\)BC#��1G�?��?�  A�z�CaH                                    ByQI�  �          @Vff?.{@'�?�B�B�Ǯ?.{@E�?^�RAw33B���                                    ByQX@  �          @X�þǮ@&ff?�(�B=qB��H�Ǯ@Fff?�  A�(�B�.                                    ByQf�  "          @q�?��H@E�?�z�A�{B�Q�?��H@]p�?\)A�RB�=q                                    ByQu�  
(          @~{@��@8Q�?ǮA�z�BU�@��@O\)?�@��Bb(�                                    ByQ�2  
�          @�G�@�H@>�R?��A�ffBK�H@�H@P��>�\)@�Q�BU�R                                    ByQ��  T          @�Q�@!�@C�
?��A�{BI�@!�@Y��>�ff@�Q�BU\)                                    ByQ�~  "          @��@.{@*�H?��
A��B1�
@.{@A�?\)@��B@(�                                    ByQ�$  "          @�{@'�@@  ?��A���BC�@'�@P  >aG�@E�BLp�                                    ByQ��  �          @�Q�@+�@J=q?�G�A[�BF�R@+�@TzὸQ쿞�RBL33                                    ByQ�p  
�          @�\)@6ff@QG�?��Ab�\BD  @6ff@\�ͽL�Ϳ333BI��                                    ByQ�  +          @��R@8Q�@P��?uAG�BB33@8Q�@Y���8Q��G�BF��                                    ByQ�  K          @��@Dz�@B�\?c�
A:�\B3  @Dz�@J=q�.{�{B7�\                                    ByQ�b  �          @�p�@E@C33?W
=A/\)B2��@E@I���aG��8��B6�                                    ByR  +          @�p�@0��@Tz�?k�A@��BIG�@0��@\(��u�G
=BM(�                                    ByR�  }          @��@(��@Z=q?uAI�BQ�@(��@a녾k��@��BU�                                    ByR%T  
�          @�p�@"�\@\��?���A`(�BVz�@"�\@g
=�\)��\)B[\)                                    ByR3�  �          @�(�@\)@^�R?p��AF�RBY�\@\)@fff��\)�fffB]{                                    ByRB�  �          @��@'
=@\(�?\(�A3�BS33@'
=@a녾�{��\)BU��                                    ByRQF  �          @��R@*�H@^{?Q�A*�RBQ�@*�H@b�\�Ǯ���RBT
=                                    ByR_�  T          @��R@'�@_\)?fffA9p�BTff@'�@e������
=BWp�                                    ByRn�  �          @��@ ��@fff?k�A>{B\\)@ ��@l�;�{���\B_Q�                                    ByR}8  �          @�@{@e?L��A&�\B]�R@{@i����ff��33B_�                                    ByR��  �          @�z�@%@Z=q?p��AE��BSG�@%@a녾���\��BV��                                    ByR��  �          @�z�@$z�@Y��?��AZ�HBS��@$z�@c33�.{��BX\)                                    ByR�*  
�          @�
=@ ��@dz�?p��AB�\B[G�@ ��@j�H���
��=qB^z�                                    ByR��  
�          @�
=@�@j=q?��A[\)BfQ�@�@s33�u�Dz�Bj=q                                    ByR�v  �          @�p�@�@hQ�?xQ�AJ{Bd��@�@o\)���R���\Bh�                                    ByR�  �          @�p�@ff@i��?\(�A4  Bd��@ff@n{��
=��{Bf��                                    ByR��  �          @�(�@   @aG�?(��A33BZ�@   @b�\����\)B[                                      ByR�h  "          @��
@�@b�\?E�A"�\B^33@�@fff����ǮB_�R                                    ByS  �          @��@,��@@  ?��RA�z�B@�@,��@P  >��@G�BH�                                    ByS�  
�          @���@%@P  >�
=@�G�BM�
@%@L�Ϳ333��BL�                                    BySZ  
c          @�z�@#�
@c�
>8Q�@��BY=q@#�
@Z=q����Z�\BT�\                                    ByS-   K          @��@{@q�>��@��HBnp�@{@l�Ϳfff�<z�Bl�                                    ByS;�  T          @�p�@p�@s�
>#�
@�Bo��@p�@h�ÿ���p(�Bk                                      BySJL  T          @�(�@(�@q�>B�\@��Bo��@(�@g���{�k�Bk�\                                    BySX�  
�          @�G�?�33@r�\>��H@�33B}��?�33@n�R�Tz��333B|(�                                    BySg�  
�          @���?�(�@u�?.{A�RB��?�(�@u��+��Q�B��=                                    BySv>  �          @���?�=q@u���
�uB�{?�=q@fff��=q��\)B|ff                                    ByS��  
�          @�33?�Q�@u=L��?0��B}{?�Q�@hQ쿡G���z�Bw                                    ByS��  �          @�=q@@dz�>�33@�
=Bb��@@^{�c�
�?33B_��                                    ByS�0  
�          @�\)@z�@|(�=�\)?fffBy(�@z�@n{���
��Bs�
                                    ByS��  
�          @�
=?�Q�@��׾u�@��B�� ?�Q�@l�Ϳ�����\)ByQ�                                   ByS�|  T          @�
=?��H@�  ������
=Bz�?��H@i����
=���\Bv�
                                    ByS�"  ]          @��R?���@��׿�\��(�B�� ?���@fff��{��\)B{
=                                    ByS��  K          @�Q�@�@}p��333�
=B{z�@�@^�R� ���ԣ�Bo                                      ByS�n  
Z          @�\)@\)@u�\)��(�Bo
=@\)@Z=q����  Bc
=                                    ByS�  �          @�ff@p�@j�H������HB`z�@p�@R�\�ٙ���(�BT�                                    ByT�  T          @�\)@�@r�\���R�\)Bh{@�@\�Ϳ˅���HB^G�                                    ByT`  �          @�\)@<(�@W����Ϳ��BC��@<(�@HQ쿡G���ffB;\)                                    ByT&  T          @�(�@4z�@U���Ϳ�G�BGff@4z�@Fff���R��B?
=                                    ByT4�  
c          @���@*=q@Q녾\)��z�BK��@*=q@A녿�G�����BC
=                                    ByTCR  �          @���@33@n{����[�Bi=q@33@Y�����
��G�B_�H                                    ByTQ�  �          @���@#33@Z=q������BT�
@#33@J=q������\BLff                                    ByT`�  
c          @��
@(��@X��=#�
?
=BPQ�@(��@L�Ϳ����s�BI                                    ByToD  K          @�Q�@G�@L(�>��?��HB6��@G�@B�\�u�F=qB1\)                                    ByT}�  
�          @��@[�@<��>��@��B"�\@[�@:=q�!G�� z�B!                                      ByT��  	�          @�(�@u@{?5A�B
=@u@$z�B�\�B�                                    ByT�6  K          @�p�@_\)@C33?!G�@�\)B$�R@_\)@E�����ffB%�
                                    ByT��  
�          @��@b�\@9��?��RAr�RB33@b�\@I��>#�
?���B&�\                                    ByT��  �          @�
=@h��@7
=?E�A�B@h��@<�;�=q�P��B=q                                    ByT�(  
�          @���@Q�@L(�>�G�@�ffB0�@Q�@I���333��B/=q                                    ByT��  
�          @�
=@g�@7�?Q�A!p�Bff@g�@>{�aG��.�RBz�                                    ByT�t  T          @�@c�
@<��>��@�=qB�@c�
@;��z���{B                                    ByT�  �          @��@g
=@1�>�?˅B=q@g
=@)���Y���+
=B
=                                    ByU�  �          @���@|��@�>��@��A�33@|��@Q�\��{A���                                    ByUf  
�          @���@p��@\)>��
@���B33@p��@�Ϳ���\)B=q                                    ByU  �          @���@^{@3�
?z�@�ffB33@^{@5��(����BG�                                    ByU-�  "          @�33@z=q@�>��R@z=qA�(�@z=q@�\������A���                                    ByU<X  T          @��@mp�@7�?#�
@�  B�@mp�@:�H�������
B                                    ByUJ�  
�          @�  @i��@=p�?�R@��Bp�@i��@@  ��ff���RB��                                    ByUY�  
�          @�  @dz�@B�\?L��A�B!@dz�@HQ쾞�R�l��B%
=                                    ByUhJ  T          @�Q�@e�@C33?0��A��B!@e�@Fff��
=��=qB#�                                    ByUv�  �          @�p�@c�
@:�H?J=qA�
B�@c�
@@�׾�\)�Z�HB!
=                                    ByU��  �          @��@`  @@��?(�@�\B"��@`  @A녾�����HB#�\                                    ByU�<  
�          @���@Y��@H��>�(�@���B*��@Y��@E�333�	��B(��                                    ByU��  �          @��H@U�@H��>�=q@Y��B-z�@U�@B�\�W
=�)��B)�                                    ByU��  �          @�Q�@QG�@E>��?�z�B-p�@QG�@<�Ϳp���@��B(
=                                    ByU�.  �          @�G�@\��@;��B�\�
=B!z�@\��@+������uBz�                                    ByU��  
�          @��@\(�@?\)�#�
�.{B#��@\(�@2�\����V�HBff                                    ByU�z  T          @�\)@H��@J�H>�{@�=qB5Q�@H��@E�J=q�#�B2G�                                    ByU�   T          @�  @C33@Q�>�(�@��RB<�@C33@N{�@  ���B:{                                    ByU��  
�          @�33@.{@W
=>.{@(�BL
=@.{@L�Ϳ��\�Z�RBF�R                                    ByV	l  "          @���@p�@h�ýu�@  Bk33@p�@X�ÿ�=q��p�Bc�H                                    ByV  ]          @��R@p�@Z�H�\(��>�RBd�@p�@9���   ��Q�BS                                      ByV&�  
�          @�p�@
=@w
=>��@[�Bu(�@
=@mp���{�h��Bq33                                    ByV5^  T          @���?�Q�@w
=?�RA33B}=q?�Q�@u�E��"�HB|��                                    ByVD  
�          @���?�
=@}p��W
=�1p�B��?�
=@Z=q����z�B~�H                                    ByVR�  �          @�ff?Ǯ@�녿Y���0z�B���?Ǯ@`���{��z�B�                                    ByVaP  "          @�G�@!G�@:=q����\BDff@!G�?�33�Mp��6{B�                                    ByVo�  T          @�?���@`  ��33����B��
?���@J=q��ff���HB��3                                    ByV~�  "          @x�ÿ=p�@a�?�A�{Bˣ׿=p�@s33=�Q�?�=qB���                                    ByV�B  
�          @u��@  @P  ?��
A��B�#׿@  @j�H?�@��RB�=q                                    ByV��  "          @L(�����@,(�?�ffA�{B�녾���@C�
?�\AB��H                                    ByV��  
�          @Fff���R@��?�(�B
=B=���R@9��?=p�Ad��B�Q�                                    ByV�4  �          @e��fff@%@(�BB�\)�fff@L(�?��A�  B�z�                                    ByV��  "          @|(��\@\)@5B9�B���\@Fff?�\)A��B��                                    ByVր  
�          @��H�  ?�z�@:�HB8�RC�{�  @%�@	��B (�C
                                    ByV�&  �          @�(��=p����@-p�B*�C6���=p�?Q�@%B �HC$��                                    ByV��  
(          @��
�C�
�(��?�z�A�=qC\���C�
���@�RB
Q�CS��                                    ByWr  	�          @���S�
�(�?�AîCUn�S�
��z�@p�B	ffCK\                                    ByW  T          @�{�Z=q�	��?���A��HCTG��Z=q��{@\)Bz�CI��                                    ByW�  	.          @���fff��
=?��A˅CM
=�fff�h��@
=B�
CB#�                                    ByW.d  
�          @��
�h�ÿ��@	��A�\CD=q�h�þk�@Q�B��C7��                                    ByW=
  �          @�\)�^�R�333@�\B\)C?ff�^�R>��@��B	�HC1}q                                    ByWK�  
�          @j�H�7�?:�H?��RB�
C%�\�7�?���?�{A�ffCG�                                    ByWZV  T          @[����@ff?��HB�RB�#׿��@9��?�  A��
B�=                                    ByWh�  T          @p  @�@{?}p�A�\)B8�@�@*=q=u?h��BA=q                                    ByWw�  �          @~{@@��@�=�Q�?��Bff@@��@33�E��8��B�                                    ByW�H  K          @�z�@{�?��?z�@�33A�ff@{�?��H���ǮA�Q�                                    ByW��  
c          @�
=@���?�Q�>L��@%A�(�@���?�녿   ��z�A��                                    ByW��  �          @��@~�R@z�=#�
?��Aۙ�@~�R?����0���A���                                    ByW�:  �          @��\@w�?�����(�A�Q�@w�?\����j{A��
                                    ByW��  T          @���@G�@0��?��A��\BJ�@G�@C�
>�\)@��BU��                                    ByWφ  	.          @��?��H@aG�?�33A�G�B��?��H@xQ�>���@��
B��                                    ByW�,  �          @���@
=q@[�?�33A��RBg{@
=q@l��=���?�=qBn�                                    ByW��  �          @�z�@	��@L(�?�\)A�G�B`�@	��@^{>#�
@��Bh�                                    ByW�x  �          @�G�@��@B�\?�\)A�B[G�@��@U�>W
=@>{Bd�H                                    ByX
  �          @~�R@33@1G�?��A��BIG�@33@H��>�@���BW
=                                    ByX�  �          @|��@�H@5?�G�A�\)BF
=@�H@Fff>.{@"�\BO��                                    ByX'j  �          @z�H@�\@Dz�?=p�A/�BT��@�\@HQ�Ǯ��  BW
=                                    ByX6  �          @�Q�@*�H@@  ���
��33BA(�@*�H@333����z�\B9(�                                    ByXD�  �          @��H@%@B�\>8Q�@$z�BFG�@%@:=q�h���R{BA33                                    ByXS\  �          @��\@&ff@]p�    <��
BTff@&ff@N�R��(���p�BM                                      ByXb  �          @�z�@(��@C�
�J=q�1�BE
=@(��@%���=q�ң�B1�\                                    ByXp�  �          @�33@	��@Vff?��@���Bd�@	��@U��+���BdG�                                    ByXN  �          @�=q?�(�@b�\>���@��Bs�?�(�@[��s33�W�
Bpp�                                    ByX��  �          @���@@e���\)���\Bo  @@TzΎ����\Bgff                                    ByX��  �          @���@\)@h��=L��?:�HBi��@\)@[����R���RBc=q                                    ByX�@  �          @��R@�\@`  ��Q�����Bc=q@�\@I��������=qBW�
                                    ByX��  �          @��R@p�@N�R�xQ��W
=BS�@p�@+��33��B>                                      ByXȌ  �          @�33@,(�@(���������B1��@,(�?��H�<(��*��Bz�                                    ByX�2  �          @��R@��@=p���\��33BK@��@���,��� �B&�
                                    ByX��  �          @��\@�@7
=�ٙ����BK\)@�@z��&ff�ffB&��                                    ByX�~  T          @�G�@{@N�R�n{�Up�B^{@{@,(��G����BJ                                      ByY$  �          @z=q@z�@G����\�r�HBb  @z�@#�
��
�   BL\)                                    ByY�  �          @~{?�z�@E�������z�Bh�?�z�@z��%�#
=BI                                    ByY p  �          @|(�?��@G���{��
=Bp�
?��@ff�'
=�&
=BR�                                    ByY/  �          @~{?�(�@L�Ϳ�����p�Bw=q?�(�@(��'
=�$�B[\)                                    ByY=�  �          @|(�?�Q�@K���{��ffBxG�?�Q�@���(Q��'Q�B[�H                                    ByYLb  �          @~�R?��@^{�����B�.?��@5����ffBu�
                                    ByY[  �          @�  ?�(�@_\)�����ffB�u�?�(�@8Q�������B|{                                    ByYi�  �          @��H?���@\���ff��z�B�(�?���@�R�L(��=�\Bf�                                    ByYxT  �          @��?�  @fff�{��B��
?�  @\)�e�T��B�#�                                    ByY��  �          @�z�?��
@s33���ڏ\B�8R?��
@3�
�U��8�\Bu                                    ByY��  �          @��@  @qG������=qBl��@  @>�R�1G��G�BT                                      ByY�F  �          @�G�@#�
@xQ쿊=q�QG�Bb(�@#�
@O\)�Q��BN                                    ByY��  �          @�Q�@Dz�@�{�O\)�G�BW{@Dz�@hQ��\)�ɅBG33                                    ByY��  �          @�Q�@.{@��Ϳn{�#33Bi(�@.{@q�����{BYG�                                    ByY�8  �          @���@#33@��ÿTz���Br�\@#33@|(��Q����HBd�                                    ByY��  �          @�ff@��@��ÿ0����\Bv\)@��@�  �  �̣�Bi�H                                    ByY�  �          @�  @#�
@����   ��ffBr�R@#�
@��\�����\Bg                                    ByY�*  �          @�Q�@!G�@�33��Q��x��Bt�@!G�@����H���Bk\)                                    ByZ
�  �          @�  @$z�@�=q�k��\)Br�\@$z�@�ff��=q��33Bi��                                    ByZv  �          @��@%�@��
?�@�ffBg�\@%�@�녿k��-��Be��                                    ByZ(  T          @��H@�R@�z�>u@8��Blp�@�R@}p����R�l��Bg�H                                    ByZ6�  T          @�(�@ff@��������B�.@ff@��
��=q��G�B|�                                    ByZEh  �          @�\)@�@�\)>���@qG�B��@�@�녿��
�l��B|��                                    ByZT  �          @�  @%�@u?z�HA>�\B`z�@%�@|(���G���z�Bc{                                    ByZb�  �          @�{@�
@|(�?���AR�\Bnp�@�
@�녾Ǯ����Bq�                                    ByZqZ  �          @��
@   @s33?L��A   Bb��@   @u�
=��Bc�H                                    ByZ�   �          @��@��@u?��A��
BnG�@��@�=q���Ϳ��\Bs�                                    ByZ��  �          @��
@�@xQ�?��RA��By�R@�@��<��
>�  B�                                      ByZ�L  �          @�@ ��@k�?�33A���Bu�\@ ��@|(�<#�
>�B{�                                    ByZ��  �          @�?�(�@g�?���A��
Bu��?�(�@|��>aG�@7�B~
=                                    ByZ��  �          @�z�?�p�@\(�?�\)AɮBp�?�p�@xQ�?�@�  B|                                      ByZ�>  �          @��
@33@`��?���A���Bn�R@33@u>k�@AG�Bwp�                                    ByZ��  �          @��?�\)@e?˅A�\)BzG�?�\)@z�H>aG�@9��B�#�                                    ByZ�  �          @�Q�@p�@u�.{�33Bp�@p�@Vff���R��z�BbG�                                    ByZ�0  �          @���@Q�@z�H���
�y�Bj�H@Q�@N{�%��=qBU��                                    By[�  �          @�
=@=q@u���
�|��Bg�\@=q@H���#33�33BR{                                    By[|  �          @�Q�@
=@xQ쿴z����Bj��@
=@H���+��	�BT33                                    By[!"  �          @���@ff@z�H��z����Bl(�@ff@J�H�,���	\)BU�
                                    By[/�  �          @�\)@\)@�녿�(���
=Bj�@\)@Q��333��HBS�\                                    By[>n  �          @�ff@\)@��׿��H��\)Bh�R@\)@P  �1��BQ��                                    By[M  �          @�G�@ ��@��H���
���Bi��@ ��@R�\�7���BR\)                                    By[[�  �          @�=q@ ��@��
�Ǯ����Bjp�@ ��@S�
�:=q�{BS{                                    By[j`  �          @�=q@��@��ͿǮ���
Bm@��@U�:�H��HBV��                                    By[y  �          @���@{@��\��ff���Bkz�@{@Q��8Q��G�BTG�                                    By[��  �          @��
@%�@���У�����Bg��@%�@QG��>{��
BO33                                    By[�R  �          @�(�@(Q�@��H�У���ffBd�H@(Q�@P���=p���BL(�                                    By[��  �          @�\)@�H@�Q���H���\Bk��@�H@J=q�@���z�BR{                                    By[��  �          @��\@�
@r�\�����
=Bj�@�
@8Q��Fff� �BMff                                    By[�D  �          @�p�@�@z�H�����
=Bk(�@�@B�\�C33�p�BP�                                    By[��  �          @���@p�@qG���Q����HBc�R@p�@6ff�H���p�BE{                                    By[ߐ  �          @��H@��@p�׿������Bfff@��@7
=�E��{BH��                                    By[�6  �          @��@�@qG���\)���RBi{@�@8Q��Dz���BK��                                    By[��  �          @��@33@xQ��{����BmQ�@33@E��7
=���BT�                                    By\�  �          @��H@z�@tz��ff��=qBv33@z�@6ff�S33�,�RBW�H                                    By\(  �          @��?���@x����\���B�L�?���@6ff�`���<=qBr��                                    By\(�  �          @��@��@u�ٙ����Bp��@��@@���;���HBW33                                    By\7t  �          @��\@=q@n{��33���Bdz�@=q@4z��E����BFQ�                                    By\F  �          @���@!G�@j=q��ff���\B^  @!G�@333�=p���
B@Q�                                    By\T�  �          @�  @'
=@Z�H�   ���BRQ�@'
=@   �Dz��!=qB/                                      By\cf  T          @���@/\)@W��33��(�BKp�@/\)@(��E� �B&��                                    By\r  T          @���?���@x���{�ظRB}33?���@7��\(��2��B^�H                                    By\��  �          @�z�?��
@����
���B�� ?��
@C33�fff�=B��{                                    By\�X  �          @�?���@�����R��
=B�\)?���@G
=�a��7(�B��                                    By\��  �          @�z�?��H@�33�\)��=qB�?��H@C�
�a��8z�B�                                      By\��  �          @���?��@��H����\)B��
?��@E��]p��:33B�W
                                    By\�J  T          @�?��
@}p��
=����B�(�?��
@>�R�Vff�7�B��                                    By\��  �          @�z�?�z�@x����֣�B�(�?�z�@:�H�S�
�6ffB�z�                                    By\ؖ  �          @�{?��R@z�H������B��?��R@8���^�R�?��B���                                    By\�<  �          @�(�?���@�G�� ����B�k�?���@E�Q��4��B��                                    By\��  �          @���?�{@{���\���B��
?�{@>�R�Q��4
=B��                                    By]�  �          @�G�?�p�@y����(���G�B�
=?�p�@>�R�L���3  B�33                                    By].  "          @���?�G�@}p�����B�� ?�G�@C33�K��2B��                                     By]!�  T          @��?E�@��
������B�?E�@Mp��L(��/��B���                                    By]0z  �          @��\?E�@����\)����B��
?E�@Mp��K��/�\B��                                    By]?   �          @��R?��@��R�����B�W
?��@G
=�n�R�B�RB�(�                                    By]M�  �          @��?+�@����#33��G�B�8R?+�@:=q�q��N��B�(�                                    By]\l  �          @��?W
=@����Q���  B���?W
=@N{�n{�@B��
                                    By]k  �          @���?.{@�ff�33�أ�B��\?.{@XQ��l(��;B�33                                    By]y�  �          @�  ?z�@�Q��z���  B�G�?z�@aG��`  �1B�8R                                    By]�^  �          @�(�>8Q�@��H��33��Q�B�z�>8Q�@o\)�HQ����B�u�                                    By]�  �          @�\)�\)@�
=���
���B�� �\)@j�H�>{���B�B�                                    By]��  �          @���\)@�Q쿹����(�B�� �\)@o\)�:=q�p�B�8R                                    By]�P  �          @��\��@�(���(��vffB�녿�@l���(���=qB�u�                                    By]��  �          @��\�   @��
��  �~�RB���   @k��+��Q�BÏ\                                    By]ќ  �          @�����
@�녿��\��G�B�8R���
@g
=�*�H��
B���                                    By]�B  �          @�=q�\)@�G��Ǯ���B����\)@`  �;����B�z�                                    By]��  �          @�녽u@�����(���33B��R�u@b�\�7
=��B��                                    By]��  �          @�G���\)@��\������p�B��὏\)@g��.{��RB�aH                                    By^4  �          @�=q>��@��Ϳ�����HB��
>��@P���L(��1Q�B��3                                    By^�  �          @��>u@x������
B�{>u@7
=�a��K��B���                                    By^)�  �          @��H>�  @~{�33���\B��)>�  @<(��aG��G�B�ff                                    By^8&  �          @��\>�G�@�����ظRB���>�G�@Fff�U�;�B��H                                    By^F�  �          @��>�G�@����	���ޣ�B��>�G�@E��Y���>ffB���                                    By^Ur  �          @���?&ff@��H�ff����B��?&ff@HQ��W
=�9�RB��                                    By^d  �          @���>���@��\��33��Q�B�p�>���@`���AG��"G�B�                                      By^r�  �          @��
?h��@����  �G�B���?h��@s�
�(����HB�{                                    By^�d  
�          @�33>�
=@�ff��G��L(�B�L�>�
=@u�p��
=B���                                    By^�
  �          @�(�>�z�@����\)��\)B�p�>�z�@k��1����B���                                    By^��  �          @�G��aG�@�{��Q����B��aG�@W��@���'{B�z�                                    By^�V  �          @�녿(�@���������B���(�@e��1G��33B�W
                                    By^��  �          @��\�\@����{�`Q�B��R�\@q��!���
B�W
                                    By^ʢ  �          @��þ�33@��Ϳ��\�P(�B�
=��33@r�\�(��B�z�                                    By^�H  �          @�\)���
@�녿��H�z=qB�\���
@i���%�z�B�p�                                    By^��  �          @�녿333@�ff<��
>���B��Ϳ333@�{��p���\)B��)                                    By^��  �          @��\�J=q@�{���
���HB�{�J=q@�=q��ff���B��)                                    By_:  �          @�33�L��@��׿O\)�"�HB�#׾L��@~�R��\���B���                                    By_�  �          @���?�R@��ÿ��\���\B�k�?�R@fff�(Q���B�aH                                    By_"�  �          @��
?�p�@u���Q���=qB�Q�?�p�@B�\�8Q��%{B��)                                    By_1,  �          @�(�?�\)@i�����
���B�
=?�\)@5�9���&33Bq
=                                    By_?�  �          @�(�@��@O\)��ff����B_(�@��@���1�� �B@(�                                    By_Nx  �          @���<#�
@�G��������B���<#�
@W
=�'����B��\                                    By_]  �          @�=q��@�=q��z���z�B�Ǯ��@W
=�,(��\)B��{                                    By_k�  �          @�z�=���@�G������
B�8R=���@P  �;��(  B��=                                    By_zj  �          @�G�@4z�@  ���H��p�BQ�@4z�?�(��&ff�A�{                                    By_�  �          @�=q@z�H?�Q���R��\)A��@z�H>�����@ָR                                    By_��  �          @�G�@N{@����
�Ǚ�Bff@N{?�{����A��                                    By_�\  �          @��H@K�@�Ϳ�z��ҸRB
�@K�?�Q��"�\�{A\                                    By_�  �          @�Q�@@��@\)��33��G�B(�@@��?�ff�Q���
A�z�                                    By_è  �          @�
=@%�@E����H��  BH  @%�@   �
=q��(�B0Q�                                    By_�N  �          @�ff@��@E�������BN  @��@��
=�	z�B3=q                                    By_��  �          @�\)@)��@:�H��(����B?{@)��@G��
=�=qB"�                                    By_�  �          @��
@8��@=q�ff��BQ�@8��?�=q�1�� A�33                                    By_�@  �          @�{@�
@:=q�ff��ffB[�@�
@33�<���5�RB3��                                    By`�  �          @��
?�=q@Y���ٙ���p�B��?�=q@)���.�R�&�Bl�                                    By`�  �          @���?��R@Dz��
=q�p�B�(�?��R@(��C33�DffB^�                                    By`*2  �          @�(�?���@l(��������B�  ?���@C�
�����B�u�                                    By`8�  �          @�?���@c33��33��(�B�z�?���@3�
�.�R�#
=Bz��                                    By`G~  �          @�ff?�  @`  ��\�Ǚ�B�k�?�  @.{�4z��(�Bt33                                    By`V$  �          @�?���@e��������
B�Ǯ?���@2�\�9���0��B�
=                                    By`d�  �          @��
?�33@`  ��������B��3?�33@-p��7��1=qB�{                                    By`sp  �          @��
?u@hQ��=q����B�k�?u@:=q�+��$��B��\                                    By`�  T          @�z�@   @Mp���33���RBhG�@   @   �'
=���BM
=                                    By`��  �          @�z�?�(�@G���\)��
=Bg{?�(�@�2�\�)z�BG�R                                    By`�b  �          @�z�?��H@N{�������Bj�\?��H@ ���&ff��RBO�H                                    By`�  �          @�
=?�(�@x�ÿL���1G�B�
=?�(�@Z=q�G�����B��\                                    By`��  �          @��?�33@S33��ff����Bpp�?�33@'
=�"�\��
BX(�                                    By`�T  �          @��H?c�
@��׿��R��ffB��?c�
@Y����R�  B��R                                    By`��  �          @�����R@��\�5��
B�33���R@w��z���
=B�(�                                    By`�  �          @�ff�#�
@�z�!G���B�(��#�
@|���G���Q�B�L�                                    By`�F  �          @��R�.{@��Ϳ����B�k��.{@\)������ffB��H                                    Bya�  �          @�녾\)@��R�k��:�\B��\�\)@z�H��
��(�B�
=                                    Bya�  �          @��L��@�\)���H�
=B�z�L��@g
=� ����B��q                                    Bya#8  �          @��ͽL��@�  �����nffB�aH�L��@j=q�(���
B���                                    Bya1�  �          @�Q�=���@|�Ϳ�
=���RB�=q=���@Q��'��z�B��                                    Bya@�  �          @���333@h�ÿ����z�B�녿333@7��7��0�\Bϙ�                                    ByaO*  �          @��
����@z�H�������B�������@Tz�����p�B�L�                                    Bya]�  �          @�����@|(��\(��D��B��3���@]p������RBÙ�                                    Byalv  �          @�33��p�@u��=q�y�B�
=��p�@R�\�  ���B��)                                    Bya{  �          @��׾��@�
=�Ǯ����B�{���@w���(���G�B��R                                    Bya��  �          @�G�=�Q�@�{�Tz��4  B��=�Q�@l���Q�����B�33                                    Bya�h  �          @�G�=u@�{�L���-�B�\)=u@mp��ff��B�#�                                    Bya�  �          @��=L��@��Q��2{B��q=L��@l���
=��B��{                                    Bya��  �          @��\?fff@�(��h���A��B��=?fff@hQ��
�H��\)B�G�                                    Bya�Z  �          @��?s33@��H�s33�K�
B��3?s33@dz�������B��                                    Bya�   �          @��\?aG�@�33����h(�B���?aG�@b�\���{B���                                    Bya�  �          @���?
=@�=q����g
=B�k�?
=@a���\��RB��f                                    Bya�L  �          @�p���@�(������B��=��@qG���  ����B��H                                    Bya��  �          @��R>�
=@��Ϳ����HB�� >�
=@p�׿�����
=B�L�                                    Byb�  �          @���=���@�(���z��~�RB�=q=���@u���=q���RB�                                    Byb>  �          @��R?�p�@p  �����n�RB�  ?�p�@N{������\B��q                                    Byb*�  �          @��R?�  @i�������m�B�Ǯ?�  @HQ��
=q���HBs(�                                    Byb9�  �          @��R@>�R@"�\��
=���RB!@>�R?��H�
=q���Bz�                                    BybH0  �          @��
?���@W
=������B�8R?���@'��1��-
=B{
=                                    BybV�  �          @�z�?�ff@]p���\)���RB��R?�ff@1��'���Bsp�                                    Bybe|  �          @�{?}p�@u��G����\B�(�?}p�@P������ffB�B�                                    Bybt"  �          @���?���@n{��\)��{B�Q�?���@A��,����\B��H                                    Byb��  �          @��H?�33@g���  ��ffB�?�33@8���2�\��BqQ�                                    Byb�n  �          @�G�?���@c�
���H�ٙ�B��{?���@1G��>{�/33B~�                                    Byb�  �          @��?�  @dz��
=q���
B�k�?�  @.�R�J=q�:{B��R                                    Byb��  �          @��R?fff@r�\�	����G�B�G�?fff@<(��N{�9\)B��                                    Byb�`  �          @��?J=q@`  �*=q�Q�B��=?J=q@ ���g
=�W�B���                                    Byb�  �          @�{>��@�p��k��?\)B�=q>��@�(���=q���B���                                    Bybڬ  �          @�논��
@���=L��?8Q�B������
@��H���\���
B��3                                    Byb�R  �          @��H=#�
@�=q=�G�?���B�
==#�
@�z῜(����B�                                      Byb��  �          @�{>�ff@�{�5���B���>�ff@qG����H����B�B�                                    Byc�  �          @��R?Y��@��ÿ�Q���{B��?Y��@Tz��5���B�.                                    BycD  �          @���?fff@�
=�(���33B���?fff@tz���иRB�W
                                    Byc#�  �          @�p�?G�@��H�u�G�B��?G�@�녿�ff���HB�Q�                                    Byc2�  �          @�z�?+�@��\�8Q��33B�p�?+�@�=q���R����B�W
                                    BycA6  �          @�(�?\)@�=q��G�����B��?\)@~�R��p����RB��
                                    BycO�  �          @�z�?@  @�녾�33��=qB�aH?@  @�  ������B��f                                    Byc^�  �          @��?Q�@�녾�����33B�ff?Q�@�Q�У���\)B���                                    Bycm(  �          @�z�?5@��\�k��=p�B�G�?5@�녿��
���RB�
=                                    Byc{�  T          @�G�?�\)@�Q�u�P(�B���?�\)@b�\�Q����B�                                      Byc�t  �          @�=q?B�\@�  ������p�B�ff?B�\@Y���!G���B��3                                    Byc�  �          @�33?O\)@���fff�@  B��H?O\)@l���
=��\)B�.                                    Byc��  �          @�=q?k�@���=p����B�?k�@o\)��������B�\)                                    Byc�f  �          @�G��aG�@��׼��
��\)B��)�aG�@�=q��ff���B�#�                                    Byc�  �          @�  >#�
@��>.{@z�B�>#�
@��H�����qp�B���                                    BycӲ  �          @�
=>�
=@�>#�
@\)B�u�>�
=@��ÿ���p��B�\                                    Byc�X  �          @�\)?��
@{����Ϳ��B�z�?��
@n{��G����B�W
                                    Byc��  �          @�
=?�\)@o\)��Q����
B}�
?�\)@^{���H��(�Bv�                                    Byc��  �          @��R@�@fff��ff�ÅBn  @�@S33��  ����Be=q                                    BydJ  �          @�ff@
�H@e���z��~�RBkff@
�H@U�����
=Bd                                      Byd�  �          @��R@ff@g
=��p���=qBo(�@ff@U������Bg33                                    Byd+�  �          @�{@Q�@Z=q��(�����B\�@Q�@HQ쿵���RBS33                                    Byd:<  �          @��?��?�(��`  �nB���?��?Y���z�H\Bd�\                                    BydH�  �          @�G�?\)?�z��`���q�B�(�?\)?J=q�z=q�BZ�                                    BydW�  �          @�Q콏\)@#�
�C33�H33B��὏\)?�ff�j�H=qB��                                    Bydf.  �          @|�ͽ�G�@7
=�.{�.ffB�=q��G�?�
=�\(��r�B�B�                                    Bydt�  �          @|�;\)@B�\�   ���B�\)�\)@��R�\�a�\B�z�                                    Byd�z  �          @|�;�\)@I������B�=q��\)@��I���U=qB���                                    Byd�   �          @x�ý�@S�
��(���33B�{��@&ff�6ff�>Q�B�33                                    Byd��  �          @s33<#�
@Z=q�����ə�B��R<#�
@2�\�!G��(ffB���                                    Byd�l  �          @u�.{@Z�H�ٙ���
=B����.{@1��'��,�HB���                                    Byd�  �          @W���R���@(�B%�C7p���R>��H@��B!
=C(Ǯ                                    Byd̸  �          @Vff�   >��
@{B%��C,��   ?s33@�\B(�CG�                                    Byd�^  �          @S33�G�?fff@ffB"�HC^��G�?�(�?�  B�
C)                                    Byd�  �          @Tz��p�?���@z�B=qCE�p�?�{?���A�z�C                                    Byd��  �          @U���?��
?���A��CY����@Q�?aG�At��C
s3                                    ByeP  �          @X���(�@  ?#�
A.=qC	\)�(�@�����C@                                     Bye�  �          @QG���@Q�?�A#33C
���@p��u���C	)                                    Bye$�  �          @HQ��=q?�p�>.{@K�C�\�=q?����������HC�                                    Bye3B  �          @a��\)@(�=�\)?�=qC�)�\)@ff�!G��&�HC��                                    ByeA�  �          @Z�H�,(�@z�>���@���Cn�,(�@zᾙ�����Cn                                    ByeP�  �          @_\)�!G�@ff=L��?c�
C	  �!G�@G���R�$z�C
                                      Bye_4  �          @\���Q�@{���
��=qC�q�Q�@
=�5�@(�C=q                                    Byem�  �          @^{��@p�=u?z�HC����@Q�#�
�)��C��                                    Bye|�  �          @dz���\@/\)���
��ffC����\@&ff�Y���\  CT{                                    Bye�&  �          @n�R�G�@<(��L���G
=B�L��G�@1G����\�~�\C^�                                    Bye��  �          @qG���\@>�R���Ϳ˅B�{��\@5��n{�f�\C ��                                    Bye�r  �          @b�\��@333������
=B�33��@'
=��=q���HC �)                                    Bye�  �          @i����@"�\?��A�p�C����@7�?E�AB�\B��H                                    Byež  �          @p����
@0  ?��A��RC����
@<��>�\)@�  C �                                    Bye�d  �          @l(��"�\@!�?c�
A_\)C.�"�\@*�H>��@��C��                                    Bye�
  �          @o\)�Q�@5�?�A ��C{�Q�@7���\)���\C�q                                    Bye�  �          @qG���R@AG����
��\)B����R@9���Y���Q�B�33                                    Byf V  �          @p���G�@;�>���@���B����G�@:=q���H��z�C �                                    Byf�  �          @p����
@E�>���@��HB����
@C33����HB�(�                                    Byf�  �          @p�׿�G�@R�\>8Q�@/\)B�Q��G�@N{�:�H�4z�B�u�                                    Byf,H  �          @tz���@G
=?(��A�HB�aH��@K��aG��W
=B�aH                                    Byf:�  �          @s�
����@Q�>�(�@�33B�\����@Q녾����B�#�                                    ByfI�  �          @p�׿�\@P��?
=A��B�\��\@R�\���
���
B�                                     ByfX:  T          @qG���33@Vff>�
=@�z�B�\��33@U��\��Q�B�R                                    Byff�  �          @p�׿�33@J=q?�\)A�B����33@Vff>aG�@Z�HB�u�                                    Byfu�  �          @u����@C33?Y��AN�RB�  ���@J�H    �#�
B���                                    Byf�,  �          @vff�\)@A�?&ffA(�B����\)@Fff�B�\�:=qB��3                                    Byf��  �          @qG���\@?\)?�G�Ay�B�����\@I��>.{@#33B��)                                    Byf�x  �          @n�R�ff@;�?h��AbffB�(��ff@Dz�=�Q�?�\)B��R                                    Byf�  �          @p  �\)@3�
?z�HAs33C ���\)@=p�>B�\@:=qB�33                                    Byf��  �          @n{�'�@��?}p�Av{C��'�@'�>���@�=qC��                                    Byf�j  �          @l���1�@33?E�AA��CW
�1�@�H>�@33C
޸                                    Byf�  T          @s�
�p�@333?8Q�A/�C\)�p�@8�ýL�Ϳ333Cs3                                    Byf�  �          @qG���@2�\?�=qA���CT{��@>�R>���@�z�B�
=                                    Byf�\  �          @u��
=@6ff?}p�Ao\)C���
=@@��>L��@@  C �                                    Byg  �          @�  �.�R@5�?:�HA'�
C�R�.�R@:�H�#�
��RC\                                    Byg�  �          @���3�
@6ff?&ffA�
C���3�
@:�H�����HC�                                    Byg%N  �          @�Q��333@6ff>B�\@.{C}q�333@333����C�q                                    Byg3�  �          @z=q�8��@%�?�@�=qC
L��8��@(Q�#�
�ffC	�                                     BygB�  �          @q��(Q�@*=q?�\@���C���(Q�@,�;L���B�\CJ=                                    BygQ@  �          @u��'
=@.{?&ffAp�C���'
=@2�\���
��33C�                                    Byg_�  �          @y���333@(Q�?
=q@�
=C���333@+��#�
���CE                                    Bygn�  �          @���/\)@8��?\)A (�Cp��/\)@<(��W
=�<��C��                                    Byg}2  �          @�z��:=q@8��>\)@   C:��:=q@5��(��(�C�
                                    Byg��  �          @�(��B�\@0��>u@VffC	�3�B�\@.�R����{C
                                    Byg�~  �          @���Fff@4z�8Q��   C	���Fff@*�H�fff�FffC.                                    Byg�$  �          @���O\)@/\)��Q����C���O\)@#�
��ff�e�C�                                     Byg��  T          @����c�
@(��0���{C���c�
@�������C�                                    Byg�p  �          @��\�J=q@4z�������C
Q��J=q@'���{�r�RCaH                                    Byg�  �          @����H��@!�?c�
AF�\C��H��@*�H>aG�@@  C�{                                    Byg�  �          @�Q��B�\@0��>Ǯ@�z�C	�B�\@1G����
��  C	�                                    Byg�b  �          @��H�U�@+�������=qC)�U�@   ����eG�C�                                    Byh  �          @�\)�fff@(�?�
=Ay��C���fff@�H?
=@��RC)                                    Byh�  �          @�{�z�H?��?��RA�ffC s3�z�H?�z�?Y��A4(�C{                                    ByhT  �          @�����p�?�=q>�?�33C&L���p�?�=q�\)��ffC&Q�                                    Byh,�  �          @�����
=?5>B�\@�C+��
=?:�H<#�
=�C*�3                                    Byh;�  �          @�=q��  ?!G�>�
=@�Q�C,��  ?8Q�>�  @G�C*�R                                    ByhJF  �          @������?0��>�ff@���C+W
����?G�>��@N�RC*=q                                    ByhX�  �          @������\?G�>L��@{C*ff���\?L��    <�C*
                                    Byhg�  �          @������?(��=�?��RC+�����?+��#�
���C+Ǯ                                    Byhv8  T          @�{���?G�=���?��HC*n���?G����Ϳ�z�C*n                                    Byh��  �          @�z�����?�z�?xQ�AAp�C%.����?�\)?(��AffC"��                                    Byh��  �          @��
���?�Q�?�{A���C$O\���?��R?��
AN�RC ��                                    Byh�*  �          @��
����?��?���A�  C"
=����?��H?���A�Q�C                                      Byh��  �          @�{�u�?��?��HA��C ��u�?�\?���A���CE                                    Byh�v  �          @����s33?��?�  A��C �3�s33?�Q�?���A�z�C�3                                    Byh�  �          @�  ���?��?�
=A�ffC&�����?�\)?��A]�C"��                                    Byh��  �          @�=q��\)?�G�?��A���C'=q��\)?���?�{AU�C#�
                                    Byh�h  �          @������>��?�  ABffC.&f���?333?\(�A&=qC+c�                                    Byh�  �          @�Q���G�?��?��\AxQ�C-{��G�?W
=?���AV�RC)��                                    Byi�  �          @����  ?�z�?ٙ�A���C$����  ?��
?���A��C =q                                    ByiZ  �          @�{�w�@�
@��AڸRC���w�@"�\?��HA�Q�C��                                    Byi&   �          @��R�s�
@@��A�p�C5��s�
@&ff?�A�C�f                                    Byi4�  �          @��R�r�\@�
@�A�RCn�r�\@%�?�\)A��C�                                    ByiCL  �          @�ff�o\)?�@'�A��C�o\)@\)@ffA�(�CY�                                    ByiQ�  �          @�ff�z=q?�Q�@   A�C�H�z=q@�R@�\A��HCL�                                    Byi`�  �          @��R��33?��\@
=A�=qC&�
��33?�p�?�A�=qC!.                                    Byio>  �          @�Q����\?��@�\A�G�C&33���\?���?�p�A�C 
                                    Byi}�  �          @�Q�����?�G�@\)A��C#.����?�ff@Q�A��
C�\                                    Byi��  �          @�G����?���@   A홚C%k����?�33@
�HA�ffC��                                    Byi�0  �          @�������?h��@�RA��C(
=����?��H@��AθRC!E                                    Byi��  �          @�����{?�z�@   A��C$�
��{?ٙ�@
�HA̸RC�                                    Byi�|  �          @�(����H?(�@)��A�G�C,����H?���@(�A�C$��                                    Byi�"  �          @��H���?+�@'�A�Q�C+����?�G�@��A��C#�q                                    Byi��  �          @�  ��
=>\@*=qB Q�C.���
=?xQ�@   A�  C'
=                                    Byi�n  �          @����  ?!G�@+�B   C+����  ?�p�@{A�C#�f                                    Byi�  T          @�  ��
=?�@'
=A�  C,�{��
=?�{@�HA�{C%L�                                    Byj�  �          @����33��\)@:=qB�C4���33?
=@6ffB�
C,O\                                    Byj`  �          @�G����\��@B�\BQ�C5�=���\?�@?\)B	��C,}q                                    Byj  �          @�{��33=��
@4z�BC2����33?5@.�RA�{C*�                                    Byj-�  �          @������<#�
@8��B�C3�f����?&ff@4z�B�C+J=                                    Byj<R  �          @�{��=q���
@7�B��C5{��=q?\)@4z�B�RC,��                                    ByjJ�  �          @�{��G��L��@:�HBC6�H��G�>�@8��B33C-�                                    ByjY�  �          @�  ��Q�?:�H@;�B��C*G���Q�?�{@,��A���C"J=                                    ByjhD  �          @�(���z�?�Q�@1�Bp�C#�R��z�?�\@��A�z�C�\                                    Byjv�  �          @�33���\?�=q@0��B  C!����\?�33@��A�z�C�R                                    Byj��  �          @��H�|��?���@333BG�C\)�|��@��@Q�A߅C�=                                    Byj�6  �          @����y��?�z�@-p�B��C��y��@p�@�\A�=qCk�                                    Byj��  T          @�Q��u?�  @+�BC� �u@�\@\)AԸRC.                                    Byj��  �          @�\)�n{@33@!�A��C#��n{@#33@G�A���C�
                                    Byj�(  T          @�\)�x��?�p�@%A�Q�C
=�x��@  @	��A��HC                                      Byj��  �          @��H�e?Ǯ@3�
BffCxR�e@Q�@��A�G�CT{                                    Byj�t  �          @�\)����@��
>�z�@p  Bݨ�����@��H�����=qB���                                    Byj�  �          @��@c�
@J=q�z���  B&\)@c�
@)���,����ffBp�                                    Byj��  �          @�z�@\)@u��4z���
=Bc�H@\)@I���dz��&�BNz�                                    Byk	f  �          @��@ ��@s�
�Fff�(�Bx��@ ��@Dz��u�9p�Bc=q                                    Byk  �          @��?�p�@Z=q�c33�'��Bo�R?�p�@%��{�P��BRz�                                    Byk&�  �          @���?��H@u�AG���B|
=?��H@G��p���6p�Bg��                                    Byk5X  T          @�G�?��@p  �P����\B��H?��@>�R�~{�C=qBk��                                    BykC�  �          @��?�@vff�J�H���B��R?�@G
=�y���=ffBo=q                                    BykR�  �          @��\?�@{��+�� �B���?�@Q��\(��,{B{�H                                    BykaJ  �          @�G�@�\@~{��ݙ�B{\)@�\@Y���G
=��Bl=q                                    Byko�  �          @�{?�Q�@���   ��
=B�k�?�Q�@vff�7
=�Q�B���                                    Byk~�  �          @�
=�^�R@�G��B�\���BȨ��^�R@��ÿ�(����
B���                                    Byk�<  �          @���33@��R�����
B�B��33@���p���0��B�B�                                    Byk��  �          @�ff�  @�zᾊ=q�J�HB�aH�  @����z��Y�B�                                      Byk��  �          @�=q�33@�
=��(���33B�p��33@�G���=q�t  B�aH                                    Byk�.  �          @�33��Q�@��þ����z�B��
��Q�@�33��\)�z=qB�B�                                    Byk��  �          @��H�ٙ�@�\)�333��
=Bۏ\�ٙ�@����33��33B݀                                     Byk�z  �          @����  @�33�L����HBݽq��  @��H�ٙ����
B���                                    Byk�   T          @�G��.{@U��33�˙�C8R�.{@6ff�*�H��C�)                                    Byk��  �          @���#�
@k���ff��(�B����#�
@S33�  ��ffB���                                    Byll  �          @���+�@h�ÿ������\B����+�@S33�G���33C
                                    Byl  �          @���"�\@y���������B�{�"�\@c33�ff��G�B�                                      Byl�  �          @��
���
@��H��  ����B�uÿ��
@z=q�$z���33B���                                    Byl.^  �          @�\)�}p�@�
=����^{BɅ�}p�@����
���B�{                                    Byl=  �          @�\)�fff@���p��{33BǮ�fff@�G��p���ffB�=q                                    BylK�  �          @�33�5@�(�?��
A=��Bď\�5@��=u?:�HB�8R                                    BylZP  �          @�Q��z�@�33>#�
?�ffB���z�@����=p��Q�B�=                                    Bylh�  �          @�33�У�@�Q�?333@�{BٸR�У�@�녾k��'
=B�aH                                    Bylw�  �          @�p���z�@�33?�\)A��Bݞ���z�@�G�?�\@���B�{                                    Byl�B  �          @�ff��Q�@�  ?.{@�z�B���Q�@�G��u�5�B�Ǯ                                    Byl��  �          @�����
@�?(�@��
BҊ=���
@��R�����_\)B�\)                                    Byl��  �          @�Q쿺�H@���?�A�p�Bٳ3���H@�
=?z�@���B�33                                    Byl�4  �          @�G���=q@���?�z�A�
=B֔{��=q@�Q�?Q�A
=B��)                                    Byl��  �          @�����  @��R?�A��B�8R��  @���?��@�(�B�=q                                    Bylπ  �          @�=q��@�ff?:�HA��B����@�  �#�
��Q�B��
                                    Byl�&  �          @�=q��p�@�z�?��AR{B�Q쿝p�@�Q�>aG�@0  B҅                                    Byl��  �          @����G�@c�
@�HA��HB�\�G�@|��?�(�A�p�B�Ǯ                                    Byl�r  �          @����@qG�@(�AӮB�  ���@��?���A���B��R                                    Bym
  �          @��H���@1G�@I��B"�C�����@S33@%�B�B���                                    Bym�  �          @��R��p�@AG�@?\)B�\B����p�@aG�@Q�A���B�                                    Bym'd  �          @�(�����@QG�@z�A�z�B�q����@e?�Q�A��B�{                                    Bym6
  �          @�
=�\)@`  ?�=qA�Q�B�W
�\)@n{?h��A=�B�{                                    BymD�  �          @�{�G�@c33?�\)A���B�B��G�@q�?s33AH��B�#�                                    BymSV  �          @��ͿУ�@j�H?��A�{B��H�У�@|(�?���Ag33B�                                    Byma�  �          @�G��33@hQ�?�G�A�\)B�Ǯ�33@x��?�=qA]�B�z�                                    Bymp�  �          @��ÿ�33@c33@G�AԸRB�.��33@w
=?���A�G�B�aH                                    BymH  �          @��Q�@P��@!G�B�\B�33�Q�@j=q?�33A��B�W
                                    Bym��  �          @����@C33@1G�BG�B�Q���@`  @(�A�
=B�\)                                    Bym��  �          @���   @aG�@(�A�Q�B�33�   @y��?��
A��RB�W
                                    Bym�:  �          @�(���?L��@�  BiQ�C �q��?�  @s33BW�C\)                                    Bym��  �          @���4z�>�z�@��\B]{C.8R�4z�?�  @~{BT(�C �=                                    BymȆ  �          @����J=q@�@G�B��C���J=q@%@-p�B�C��                                    Bym�,  �          @��S�
?�33@E�BC0��S�
@��@-p�B  C�                                    Bym��  �          @����N�R@�\@5�B{C���N�R@0  @��A�33C��                                    Bym�x  �          @���[�@�@)��B��CxR�[�@'
=@\)A�C��                                    Byn  �          @����>�R?�p�@H��B)�\Cٚ�>�R@\)@333B�C                                    Byn�  �          @����,��?���@j�HBJz�Cn�,��?���@X��B6
=C\                                    Byn j  �          @�Q��*�H?���@_\)B<�C�
�*�H@Q�@HQ�B$��C
T{                                    Byn/  �          @�  �-p�?�p�@`  B=�\C}q�-p�@�\@I��B&p�CǮ                                    Byn=�  T          @��
�Q�?�ff@aG�BF�C�{�Q�@�@J=qB-33C!H                                    BynL\  �          @�����@�@XQ�B>�CW
���@-p�@>{B!�C
                                    Byn[  �          @��R���@e�?���A�B��3���@u?���Ak�B��                                    Byni�  �          @�(����@o\)?�ffA��B�33���@y��?+�A��B���                                    BynxN  �          @����"�\@fff?uAD(�B��=�"�\@l��>�33@�
=B��                                    Byn��  �          @�Q��(Q�@fff?(�@�  B�G��(Q�@i��<#�
=���B��                                    Byn��  �          @�  � ��@(�@@  B'�C
�f� ��@)��@&ffB�
Cs3                                    Byn�@  �          @����'�@&ff@'�Bz�C0��'�@?\)@
=qA�Q�C.                                    Byn��  �          @�p��   @��@O\)B/p�C	���   @0  @5�B{CB�                                    Byn��  �          @�33�#33?�33@W�B?�HC)�#33@
�H@C�
B)�C��                                    Byn�2  �          @��
�:=q?�{@B�\B%G�Cp��:=q@z�@,��BCn                                    Byn��  �          @�33�$z�?�\)@Q�B7�\C���$z�@�@<(�B 33C	J=                                    Byn�~  �          @�G��ff?��@g
=BUp�C33�ff?���@W
=BA33C��                                    Byn�$  �          @��H�8��?�{@S�
B8=qC�3�8��?�\)@C�
B&z�C\                                    Byo
�  �          @�{�Q�?��R@XQ�B,C�\�Q�@   @G
=B��C�
                                    Byop  �          @�Q��Y��?���@U�B&�RC(��Y��@z�@B�\B�RC�
                                    Byo(  �          @�  �U?�\)@>�RB  C&f�U@�
@,(�B	�RCQ�                                    Byo6�  �          @����XQ�?޸R@(��B{C���XQ�@Q�@A��HC޸                                    ByoEb  �          @��
�i��@ ��?��A�\)C0��i��@�?z�HAP(�C�                                    ByoT  �          @�ff�L(�?�Q�@8Q�BffC�R�L(�@
=@#33B�C��                                    Byob�  �          @���S�
@z�@�RB�C  �S�
@�@Q�Aۙ�C�\                                    ByoqT  �          @�=q�Z�H@G�@\)A�
=CY��Z�H@?�33A�p�C�
                                    Byo�  �          @���fff@�?���A�Q�C��fff@�?�p�A��C�f                                    Byo��  �          @�ff�p  @�
>�(�@��HC\)�p  @ff=��
?��
C�                                    Byo�F  �          @�33�]p�@'����
����C��]p�@%������\)CaH                                    Byo��  �          @���\��@��>���@�ffC�
�\��@�R=#�
>��C:�                                    Byo��  �          @����0  @.�R?���A�\)C8R�0  @<(�?�
=A�z�C!H                                    Byo�8  �          @����AG�@:=q?�33A�  C
�AG�@G�?��HAxQ�C{                                    Byo��  �          @��
�2�\@;�@p�A�RC�f�2�\@N{?�  A��HC�3                                    Byo�  �          @�(���R@(�@K�B/p�CQ���R@8Q�@2�\BB�ff                                    Byo�*  �          @���0  ?��@Mp�B/�C}q�0  @�@9��B=qC��                                    Byp�  �          @�p��8Q�@33@(��B�C���8Q�@=q@z�A��C�                                    Bypv  �          @�z��5�?�p�@,��B
=C
�5�@@��B{Ch�                                    Byp!  �          @���@��?�33@N�RB4p�C  �@��?���@A�B&��C�                                    Byp/�  �          @�=q�Dz�?��
@Mp�B3{C!p��Dz�?�(�@A�B&CQ�                                    Byp>h  �          @�ff�HQ�?�\)@=p�B&�
C Q��HQ�?\@1G�B33C�                                    BypM  �          @�p��E?�@:�HB%��CY��E?Ǯ@.�RBC0�                                    Byp[�  �          @�Q��J�H?�
=@>�RB%�C���J�H?�=q@2�\BC��                                    BypjZ  �          @����7�?��@C�
B0p�Cn�7�?�(�@6ffB!�C�                                    Bypy   �          @�(��@  ?�\)@>�RB+�RC�{�@  ?\@333B��C.                                    Byp��  �          @����A�?���@?\)B+\)C ��A�?�  @3�
B��C�R                                    Byp�L  �          @��R�L(�?s33@=p�B&33C#^��L(�?���@333B�\C�                                    Byp��  �          @�{�^{?z�H@@  B33C$J=�^{?���@5B33CaH                                    Byp��  �          @��\�e�?�G�@Dz�B33C$5��e�?�@:=qB33Cff                                    Byp�>  �          @��H�g�?��\@C33Bp�C$J=�g�?�@8��B�\C�
                                    Byp��  �          @�33�p  ?E�@>�RBz�C(k��p  ?�z�@6ffB�
C"�{                                    Bypߊ  �          @�ff�S33?z�H@L(�B+
=C#s3�S33?�33@A�B �C
                                    Byp�0  �          @���H��?��@HQ�B-�
C!���H��?�Q�@>{B"�CO\                                    Byp��  �          @�z��O\)?���@H��B*G�C!�\�O\)?�(�@>{BQ�C�f                                    Byq|  �          @���N�R?k�@N{B/=qC$��N�R?��@E�B%\)C��                                    Byq"  �          @�33�N�R?^�R@I��B-  C$޸�N�R?��
@@��B#��CxR                                    Byq(�  �          @����XQ�?L��@Q�B-\)C&���XQ�?�(�@I��B$�C :�                                    Byq7n  �          @��H�S�
?��@G�B+p�C)�S�
?}p�@AG�B$�C#J=                                    ByqF  �          @�p��QG�>�@<(�B&�C+���QG�?Y��@7
=B!{C%^�                                    ByqT�  �          @��R�U�>��@;�B$ffC,��U�?G�@6ffB=qC&��                                    Byqc`  �          @�
=�Z�H>�p�@5B{C-���Z�H?8Q�@1G�B�C(�                                    Byqr  �          @���_\)>�@1�B(�C,#��_\)?Tz�@,��B�HC&��                                    Byq��  �          @�
=�c�
?��@(��B��C*z��c�
?k�@"�\B
�
C%z�                                    Byq�R  �          @�
=�Mp�?#�
@B�\B+ffC(���Mp�?�G�@;�B$Q�C"�
                                    Byq��  �          @�\)�S33?c�
@8��B C$���S33?��R@0��B{CxR                                    Byq��  �          @�\)�W
=?B�\@6ffBp�C'W
�W
=?���@/\)B��C!�f                                    Byq�D  �          @���^{?5@0��B��C(���^{?��@)��B�
C#aH                                    Byq��  �          @�\)�[�?=p�@1G�BG�C'Ǯ�[�?���@*=qB(�C"��                                    Byqؐ  �          @���W�?Tz�@6ffB�RC&&f�W�?�@.�RB�
C �H                                    Byq�6  �          @�Q��W�?:�H@9��B ffC'�
�W�?���@2�\BQ�C"h�                                    Byq��  �          @����U�?:�H@=p�B$  C'���U�?�=q@7
=B�HC"{                                    Byr�  �          @����P��?B�\@B�\B(��C&���P��?�{@;�B!G�C!&f                                    Byr(  �          @����K�?c�
@EB,�C$aH�K�?�  @>{B#�C��                                    Byr!�  �          @��\�K�?�G�@H��B-�C"aH�K�?�\)@@  B#�RC��                                    Byr0t  �          @�G��I��?n{@HQ�B.p�C#�)�I��?��
@@  B%��C�
                                    Byr?  �          @����L(�?n{@E�B+p�C#���L(�?��
@<��B"��C33                                    ByrM�  �          @����H��?s33@G
=B-�
C#{�H��?�ff@>�RB$�Cn                                    Byr\f  �          @�G��Dz�?k�@N{B4z�C#J=�Dz�?��
@EB+��CT{                                    Byrk  �          @�=q�A�?c�
@S�
B9�\C#�H�A�?�G�@L(�B0�
Cn                                    Byry�  �          @��H�C�
?�@W�B=33C)^��C�
?s33@Q�B6��C"�q                                    Byr�X  �          @�=q�J=q?(��@N{B3�
C(33�J=q?��\@HQ�B-G�C"+�                                    Byr��  �          @�=q�A�?+�@UB<ffC'��A�?��@P  B5z�C!�                                    Byr��  �          @���A�?!G�@_\)BA��C(.�A�?��\@Y��B;  C!xR                                    Byr�J  �          @����Dz�>�@]p�B@\)C+{�Dz�?\(�@X��B:��C$^�                                    Byr��  �          @��H�A�?@  @UB;�C&��A�?�{@O\)B4�\Cٚ                                    Byrі  �          @��R�H��?�ff@8Q�B!  C}q�H��?�{@.{B�\C�                                    Byr�<  �          @�\)�C�
?��@<��B%�Ck��C�
?�33@2�\B{C��                                    Byr��  �          @��H�C33?�p�@<��B =qCn�C33@�\@0  BQ�CE                                    Byr��  �          @�
=�Mp�?���@:�HB�C�q�Mp�@	��@,��B��C!H                                    Bys.  �          @����W
=@@8Q�B��C{�W
=@��@(��B�C�H                                    Bys�  �          @�
=�QG�@  @?\)B(�CxR�QG�@#33@/\)B�HC
=                                    Bys)z  �          @�
=�Z�H@{@6ffB�RC��Z�H@   @&ffA�=qC�                                    Bys8   T          @����e@��@/\)B=qC���e@{@   A��C�                                     BysF�  �          @��U@��@+�B�\C���U@-p�@=qA�=qC��                                    BysUl  �          @����N{@"�\@+�B�RC�q�N{@333@=qA�p�C
�q                                    Bysd  �          @��\�Z�H@(Q�@+�A��HCu��Z�H@8��@��AᙚC�
                                    Bysr�  �          @��\�X��@#33@1G�B�RC��X��@4z�@   A��CJ=                                    Bys�^  �          @�Q��^�R@��@+�BC�=�^�R@)��@�A�\C                                    Bys�  S          @�Q��mp�@p�@"�\A�C=q�mp�@��@33A�G�C�\                                    Bys��  �          @�Q��j�H@(�@'
=A�G�C5��j�H@�@Q�A�
=Cp�                                    Bys�P  �          @�p��c33@
=q@(Q�B(�C���c33@=q@��A�  C�f                                    Bys��  T          @�p��X��@	��@5B33C���X��@�H@'�B �C��                                    Bysʜ  �          @�
=�aG�@Q�@0��Bp�C�aG�@��@"�\A���C�=                                    Bys�B  T          @�=q�j�H@��@   A��C5��j�H@+�@��A�G�C�\                                    Bys��  �          @�{�\��@��@&ffA��
CO\�\��@(Q�@
=A�  C��                                    Bys��  �          @�(��S33@   @$z�A�
=C��S33@.�R@�A�  Cn                                    Byt4  �          @���W�@'
=@/\)B�HC33�W�@7
=@�RA�z�C��                                    Byt�  �          @����Z=q@#33@*�HB ffC:��Z=q@2�\@�HA�Q�C�q                                    Byt"�  �          @��\�\(�@Q�@=qA�p�CT{�\(�@%@�A�ffC�q                                    Byt1&  �          @�{�Z=q@!G�@ ��A���C��Z=q@/\)@��AڸRC33                                    Byt?�  �          @�
=�L(�@,��@+�B�HC�\�L(�@;�@�A��C	n                                    BytNr  �          @�ff�J�H@.�R@-p�B��C@ �J�H@=p�@��A�C�f                                    Byt]  �          @��R�dz�@333?��RA���C���dz�@>{?�p�A��C@                                     Bytk�  �          @��R�^{@;�?��RA��
C��^{@Fff?�(�A��RC
33                                    Bytzd  T          @�  �]p�@A�?��HA�{C
�=�]p�@L��?�
=A��RC	O\                                    Byt�
  T          @�
=�e�@5?�Q�A��\C���e�@@  ?�Q�A��HC�                                    Byt��  T          @��\(�@+�@�A���C\�\(�@8Q�@�\AÙ�C#�                                    Byt�V  �          @����k�@)��@��AɮC=q�k�@5�?�33A��CxR                                    Byt��  �          @�  �p��@&ff@33A�
=CB��p��@1G�?�A��
C��                                    Bytâ  �          @��R�vff@�H?��RA���C��vff@%?�\A�\)C{                                    Byt�H  �          @����ff@p��@�HA�33B�
=��ff@}p�@AͮB��
                                    Byt��  �          @��H����@{�@�HA�Q�Bܣ׿���@��
@�AˮB��f                                    Byt�  �          @�p��ff@L��@=qA�Q�B����ff@X��@Q�A���B�p�                                    Byt�:  �          @�  �vff@@��A�Cz��vff@�\@{A���CO\                                    Byu�  �          @��R�z=q?�z�@Q�A�C��z=q@ff@{A�Q�C�3                                    Byu�  �          @�ff�xQ�?�(�@ffA��C  �xQ�@
=q@�A�G�Cٚ                                    Byu*,  T          @�{�y��?�(�@�\A�33C(��y��@
=q@�A˙�C
                                    Byu8�  �          @��
�x��?�33@�RA�z�C�q�x��@z�@z�AɅC�                                    ByuGx  �          @��\�s33@�@
�HA�{C���s33@��@   A�{C�R                                    ByuV  
�          @�  �p  ?��R@�A��C��p  @	��?���A�(�C&f                                    Byud�  T          @�  �s33?�z�@
=A��CQ��s33@z�?���A�=qCh�                                    Byusj  �          @����u?�ff@(�A���C޸�u?�(�@�\A��HCٚ                                    Byu�  �          @�
=�tz�?�p�@
�HA��HC�)�tz�?�33@�A�p�C�
                                    Byu��  �          @��
�q�?˅@�A�z�C.�q�?�  ?��RA�{C&f                                    Byu�\  T          @�=q�q�?��R@ffA�33CxR�q�?�33?�p�A͙�Cn                                    Byu�  �          @����u?�@A�  C#
=�u?���@  A��C ��                                    Byu��  �          @���w
=?��
@G�A��
C!���w
=?���@
�HA�(�Ch�                                    Byu�N  �          @�(��tz�?�=q@G�A�  C �)�tz�?��R@
=qA�{C�f                                    Byu��  T          @���n{?�(�@��A���C!���n{?���@ffAߙ�C��                                    Byu�  �          @�ff�l��?���@z�A��
C p��l��?�(�?��HA��
Ch�                                    Byu�@  �          @�ff�dz�?˅@Q�A�\C�R�dz�?޸R@   A�=qC��                                    Byv�  �          @�{�e�?�ff@
=A���C���e�?ٙ�?��RA�
=C�                                    Byv�  �          @�ff�mp�?\?�A�Q�C���mp�?�z�?�ffA�33C�R                                    Byv#2  �          @���vff?�\)?���A���C Y��vff?�  ?�  A�\)C��                                    Byv1�  �          @���vff?�
=?���A��HC��vff?�ff?��HA�
=C�                                    Byv@~  �          @�  �xQ�?�z�?��A��C�R�xQ�?��?�
=A��CaH                                    ByvO$  �          @�Q���  ?�{?�=qA��C!.��  ?�(�?�p�A��C�\                                    Byv]�  �          @�Q��}p�?�\)?�z�A��C ���}p�?�p�?ǮA�=qCu�                                    Byvlp  �          @�  �z=q?�
=?ٙ�A�\)C޸�z=q?�ff?˅A�{Ch�                                    Byv{  �          @����xQ�?�p�?�\A��RC
�xQ�?���?�A��C�{                                    Byv��  �          @���u�?��?�G�A�=qC��u�?�z�?�33A�(�C�
                                    Byv�b  T          @����z�H?�ff?�z�A�Cff�z�H?�z�?�ffA�  C
=                                    Byv�  �          @��\�z=q?���?�G�A�
=C�q�z=q?��H?�33A���CQ�                                    Byv��  
�          @���tz�?�  ?�\A���Cc��tz�?�{?�33A��C�q                                    Byv�T  �          @��\�w
=?�
=?��A�Q�Cs3�w
=?�ff?�
=A��
C�                                    Byv��  �          @�G��mp�@ ��?�
=A��C� �mp�@�?��A�
=C@                                     Byv�  �          @����mp�?��R?�
=A�  C��mp�@ff?�ffA���C��                                    Byv�F  �          @�G��qG�?�
=?��A�G�C޸�qG�@�?�G�A��C��                                    Byv��  �          @����vff?���?ǮA�Q�CT{�vff?���?�Q�A�\)C0�                                    Byw�  �          @����p��?�Q�?��A��
C�3�p��@�\?\A�(�C��                                    Byw8  �          @����s33@�\?�Q�A��C���s33@Q�?��A���C��                                    Byw*�  �          @�33�u?�(�?�=qA���C�\�u@z�?���A�p�C�3                                    Byw9�  T          @��
�s�
@�\?�\)A��\C���s�
@��?��RA��HC�3                                    BywH*  T          @�p��vff@��?�(�A��\CO\�vff@�?�=qA�ffC\)                                    BywV�  �          @�G��e@��?���A�  CL��e@{?�Ap��CxR                                    Bywev  �          @�Q��dz�@�?��\A���C���dz�@ ��?�\)AeC�3                                    Bywt  
�          @�=q�j=q@(�?�(�Aw\)CG��j=q@ ��?���AXz�C��                                    Byw��  �          @��
�k�@ ��?��HAs\)C���k�@%�?��ATQ�C��                                    Byw�h  �          @����b�\@$z�?���Ab=qC�R�b�\@(Q�?s33AB=qCT{                                    Byw�  �          @����h��@�H?�p�A{�Cn�h��@\)?��A]��C�3                                    Byw��  �          @��R�h��@33?�
=AuG�C��h��@
=?��AX��C�                                    Byw�Z  �          @�\)�hQ�@Q�?�{Ae��C���hQ�@(�?xQ�AHQ�C{                                    Byw�   �          @���i��@��?�ffAXQ�C�i��@��?h��A;\)C&f                                    Bywڦ  �          @�  �g
=@�R?�G�AO\)C���g
=@!�?^�RA1C                                      Byw�L  �          @�\)�g
=@   ?c�
A733CQ��g
=@"�\?@  AC�{                                    Byw��  �          @���i��@�H?��\AR�\Cp��i��@{?c�
A6{C޸                                    Byx�  �          @�G��p��@
=?xQ�ADz�C�f�p��@=q?Tz�A)p�C\)                                    Byx>  �          @���o\)@   ?B�\AffC33�o\)@"�\?!G�@�z�C��                                    Byx#�  �          @����j=q@#�
?.{A
�HC
=�j=q@%?
=q@�z�C�3                                    Byx2�  �          @�\)�hQ�@#33?�R@�{C�H�hQ�@%�>�@��C��                                    ByxA0  �          @��c33@%�?��@��
C�c33@&ff>��@�=qC�                                     ByxO�  �          @��dz�@=q?�G�AT(�C�3�dz�@p�?c�
A8��Cff                                    Byx^|  �          @�\)�n�R?�  ?�
=A�\)C�)�n�R?�?˅A���C�=                                    Byxm"  �          @�\)�k�?�  ?��
A�z�C�=�k�?�?�Q�A�{Ch�                                    Byx{�  �          @��dz�@�?�=qA�{C^��dz�@
=?�(�A�=qCk�                                    Byx�n  �          @�33�c33@ ��?���A��RC� �c33@�?���A���C�H                                    Byx�  T          @��
�dz�?���?˅A��HCff�dz�@�?��RA���Cp�                                    Byx��  �          @�p��a�?�p�@G�A��
C�3�a�?�=q?�
=AυC��                                    Byx�`  �          @�p��hQ�?�=q?�(�A��C@ �hQ�?�z�?�\)A���C5�                                    Byx�  �          @�p��l(�?�G�?�33A�{CxR�l(�?�?���A�(�CxR                                    ByxӬ  �          @����n�R?�z�?У�A�=qC�R�n�R?޸R?�ffA��HC��                                    Byx�R  �          @�z��p  ?���?�{A���C�)�p  ?�
=?��
A��C�H                                    Byx��  �          @���n{?�\)?�=qA�ffCxR�n{?�Q�?�  A�p�C�                                    Byx��  �          @����fff?�{?�33A��HC�R�fff?�?��A��RC��                                    ByyD  �          @��
�h��?�ff?�A�Q�C��h��?��?�p�A���C��                                    Byy�  �          @����g�?��R?��RA�Q�C#��g�@�
?��A��
CT{                                    Byy+�  �          @����]p�@33?��
A_�Ck��]p�@?k�AG�C�f                                    Byy:6  �          @��\�Tz�@2�\<�>�ffC���Tz�@1녽�Q쿠  C��                                    ByyH�  T          @�p��U@7
=��33��Q�CaH�U@5���ƸRC�\                                    ByyW�  T          @�33�9��@7���
=��
=CG��9��@333�Ǯ���C��                                    Byyf(  T          @���)��@<�Ϳ�{���C���)��@8Q�޸R��Q�C��                                    Byyt�  �          @���  @:�H��
��{B�#��  @4z��(���(�C }q                                    Byy�t  �          @����2�\@:=q��Q���p�Cٚ�2�\@5�������C��                                    Byy�  �          @�=q�2�\@@�׿��\���
C���2�\@=p���z���z�CaH                                    Byy��  �          @��\�(��@N{�����r�\CO\�(��@J�H���\����C                                    Byy�f  �          @����9��@Dz����  CG��9��@C33�(����C�                                     Byy�  �          @���AG�@=p�����{C�{�AG�@<�;�  �Tz�C�f                                    Byy̲  �          @�\)�S33?���?�(�A��
C^��S33@ ��?���A��
C��                                    Byy�X  �          @��H�dz�?���?�\)A�z�C���dz�?�
=?�ffA�ffC�=                                    Byy��  �          @�z��e?�(�?�A�Q�CaH�e?�ff?�\A��Ck�                                    Byy��  �          @����b�\@   ?�A�ffC���b�\@z�?�=qA�
=C�3                                    ByzJ  �          @����c�
@�?�p�A�C:��c�
@�?�33A�(�C�=                                    Byz�  I          @��
�b�\@
=q?�{A���C�H�b�\@p�?��\A��C�                                    Byz$�  
�          @�Q��Z=q@�\?��Axz�C#��Z=q@�?�ffAd  C��                                    Byz3<  
          @��
�P  @
=?Tz�A:�RC��P  @��?=p�A%G�C��                                    ByzA�  
�          @�G��Fff@{>�p�@��C���Fff@�R>�=q@}p�C\)                                    ByzP�  �          @�Q��K�@�?�A�z�C:��K�@
�H?��A|��C��                                    Byz_.  "          @~�R�8Q�@/\)=�G�?�ffC}q�8Q�@/\)    <�Cu�                                    Byzm�  T          @\)�0��@333?   @�\)C�
�0��@4z�>Ǯ@��Ck�                                    Byz|z  �          @{��Mp�?�z�?�
=A�  CJ=�Mp�?���?���A�
=C�R                                    Byz�   
�          @u�Tz�?�p�?O\)AC�
Cc��Tz�?�G�?@  A3�C                                      Byz��  "          @y���N{?��
?�A�\)C(��N{?���?�{A���C�{                                    Byz�l  
�          @u�4z�@��?5A-C�H�4z�@�H?�RA�CW
                                    Byz�  
�          @tz��   @Fff�n{�aG�B��   @C�
����}p�B�\)                                    ByzŸ  I          @o\)�@:�H�W
=�S33B�B��@8�ÿs33�n=qB��)                                    Byz�^  �          @qG��+�@   >�p�@��C	  �+�@ ��>�\)@���C�)                                    Byz�  
�          @s�
�<��@��?Y��AN�RC���<��@�\?E�A:�RCE                                    Byz�  
�          @s�
�9��@Q�?+�A!p�C���9��@��?z�A��CW
                                    By{ P            @vff�5�@$z�>�ff@�p�C	�=�5�@%�>�Q�@��C	�H                                    By{�  �          @vff�7
=@"�\>�p�@��\C
L��7
=@#�
>�\)@�  C
+�                                    By{�  -          @u�4z�@{?G�A:�HC
�)�4z�@\)?0��A&{C
��                                    By{,B            @x���C33@z�?+�A\)C��C33@ff?
=AQ�Ck�                                    By{:�  
�          @xQ��Fff@p�?8Q�A*=qC�\�Fff@�R?#�
A(�CG�                                    By{I�  �          @xQ��Fff@\)?(��AG�C5��Fff@��?
=A33C��                                    By{X4  
�          @z=q�Fff@�R?Tz�ADQ�CL��Fff@  ?@  A2ffC��                                    By{f�  T          @xQ��G
=@
=?s33Ab�\C��G
=@��?aG�AQp�Cc�                                    By{u�  	�          @q��G�?�(�?W
=AMp�C���G�?��R?G�A=G�CxR                                    By{�&  "          @r�\�>�R@�?n{Ad��C}q�>�R@	��?\(�AS\)C�                                    By{��  �          @w
=�:�H@��?:�HA-�C�)�:�H@�H?&ffA33CY�                                    By{�r  �          @|���U@�?�@�\)C���U@33>��@أ�CxR                                    By{�  "          @~�R�[�?�Q�?
=q@��C���[�?���>�@޸RCT{                                    By{��  �          @|(��W�?�(�?��A ��C�q�W�?�p�>��H@�z�C�=                                    By{�d  �          @�Q��`  ?�?5A$  C�q�`  ?�=q?(��A�HCu�                                    By{�
  �          @|���^�R?�Q�?E�A5�C��^�R?�(�?8Q�A(��C�q                                    By{�  �          @~{�b�\?���?J=qA6�HC���b�\?�\)?=p�A+33C^�                                    By{�V  
�          @\)�fff?��?B�\A/33C�\�fff?���?5A$(�C��                                    By|�  T          @~{�g
=?��?fffAQ�C��g
=?�?\(�AG
=C��                                    By|�  T          @�G��g
=?�Q�?�{A}C=q�g
=?�(�?��As�C��                                    By|%H  �          @���j�H?�{?��RA��HC���j�H?��?���A�(�C0�                                    By|3�  _          @���k�?�  ?���A�{C!G��k�?��?��
A��C �q                                    By|B�  "          @����e?�?���A�  C!�R�e?��H?��A��
C!ff                                    By|Q:  	�          @��\�k�?�=q?���A��RC#�H�k�?�\)?���A��HC#\                                    By|_�  {          @����i��?}p�?���A��HC$�\�i��?��
?���A�\)C$@                                     By|n�  T          @�Q��mp�?s33?��A��HC%�3�mp�?z�H?�{A���C%:�                                    By|},  	�          @����p  ?xQ�?��Atz�C%u��p  ?�  ?��
AmC%
=                                    By|��  
�          @~�R�j�H?:�H?�\)A���C(���j�H?E�?���A�Q�C()                                    By|�x  T          @\)�c�
?^�R?�
=A���C&@ �c�
?h��?�z�A��C%�f                                    By|�  
3          @����[�?.{?�(�A�p�C(�
�[�?:�H?���A��HC'�R                                    By|��  
�          @}p��W�?:�H?�Q�A�33C'�
�W�?G�?�A�z�C&��                                    By|�j  �          @}p��Vff?!G�@   A���C)Y��Vff?.{?�p�A��\C(s3                                    By|�  �          @{��K�?\)@\)B{C)��K�?�R@�RB	��C(��                                    By|�  "          @{��O\)>Ǯ@��B��C-\�O\)>�ff@�B  C,�                                    By|�\  �          @|���J�H>�p�@�
B��C-W
�J�H>�(�@33B�
C,=q                                    By}  �          @}p��AG�>�Q�@(�B33C-(��AG�>�(�@�BffC+��                                    By}�  T          @tz��:=q?�@=qBC)���:=q?
=@��B��C(��                                    By}N  �          @q��:=q?(�@ffB��C(33�:=q?+�@BQ�C'�                                    By},�  
�          @fff�B�\?�  ?��HA�C�H�B�\?��?�
=A�33C�                                    By};�  �          @c�
�@��?���?�{A���C\)�@��?���?���A���C�\                                    By}J@  �          @c�
�B�\?��H?�33A�
=CG��B�\?�  ?�\)A���C��                                    By}X�  
�          @e��B�\?�z�?��RAģ�C��B�\?���?��HA�ffCp�                                    By}g�  
�          @g��E�?��\?�33A��C�H�E�?�ff?�\)A�p�C{                                    By}v2  "          @j=q�J=q?�p�?�\)A�z�CǮ�J=q?�G�?��A�=qC@                                     By}��  
(          @j=q�L(�?���?�{A�ffC xR�L(�?�?�=qA��\C�                                    By}�~  �          @h���@  ?0��?�A���C&�q�@  ?=p�?�33A�Q�C&#�                                    By}�$  �          @dz��B�\?�R?�\A��C(�=�B�\?(��?�G�A뙚C'�                                     By}��  
�          @W��:�H?(�?ǮA�\)C(:��:�H?&ff?�ffA���C'��                                    By}�p  
�          @S33�6ff>��H?˅A��C*&f�6ff?�?���A�  C)ff                                   By}�  
�          @W
=�8��?z�?���A�p�C(���8��?�R?�=qA�33C'��                                   By}ܼ  
�          @W
=�>{?=p�?�33A�Q�C%���>{?E�?���A���C%aH                                    By}�b  _          @Y���?\)?!G�?�(�A��
C(��?\)?+�?��HA˅C'n                                    By}�  
�          @S33�8��?�?�p�A�
=C(��8��?��?��HA��HC(B�                                    By~�  �          @Vff�:�H>���?˅A��HC,(��:�H>�G�?�=qA�\)C+p�                                    By~T  I          @\(��?\)>��?�Q�A�p�C.��?\)>���?�
=A�ffC.0�                                    By~%�  �          @c�
�Dz�>�  ?�ffA�(�C/c��Dz�>�z�?��A�G�C.��                                    By~4�  "          @c33�C33>��
?��
A���C-��C33>�p�?�\A�C-.                                    By~CF  "          @`  �A�>�33?�p�A���C-Y��A�>Ǯ?�(�A�C,�)                                    By~Q�  "          @c33�C33>���?��
A��C.s3�C33>�{?�\A�  C-�3                                    By~`�  T          @`  �@��>��?�G�A��C/  �@��>���?�G�A���C.@                                     By~o8  T          @l(��J=q=#�
?�z�A���C3T{�J=q=��
?�z�A���C2��                                    By~}�  �          @k��J�H�L��?�\)A�{C4��J�H�#�
?�\)A�(�C4+�                                    By~��  �          @l���L�ͽ�\)?�{A�C55��L�ͼ�?�{A��
C4s3                                    By~�*  �          @n�R�J=q��=q?��HA�p�C8�f�J=q�k�?��HA�=qC8)                                    By~��  �          @]p��=p�=���?�  A��RC2  �=p�>\)?�  A�=qC1=q                                    By~�v  �          @J�H�+�?!G�?ǮA�\C&�)�+�?(��?�ffA�(�C&(�                                    By~�  
�          @J=q�(��?��?�\)A���C(33�(��?
=?���A�p�C'u�                                    By~��  T          @N{�1G�>��?У�A��C0��1G�>8Q�?У�A�\)C0.                                    By~�h  T          @Q��3�
    ?�Q�A�C3�3�3�
=#�
?�
=A��C30�                                    By~�  	�          @P���1�<#�
?��HA�Q�C3���1�=L��?ٙ�A�(�C2�                                    By�  �          @W��7
=�aG�?�G�A��RC8z��7
=�B�\?�\A�p�C7��                                    ByZ  �          @X���8Q�8Q�?�\A��\C7�
�8Q�\)?��
A��C6��                                    By   �          @W
=�8Q����?�(�A�Q�C6��8Q�u?�(�A��\C5E                                    By-�  T          @U�9��    ?�z�A��HC4
=�9��<�?�z�A��HC3T{                                    By<L  �          @U�8��=�G�?�
=A�
=C1ٚ�8��>\)?�A��C1!H                                    ByJ�  �          @W��7
==�\)?��
A��HC2��7
==�G�?��
A��\C1�                                     ByY�  �          @W��:�H=�\)?�Q�A��C2���:�H=���?�Q�A���C1�R                                    Byh>  �          @W��;�=#�
?�A�p�C3L��;�=�\)?�A�G�C2��                                    Byv�  �          @W��:=q<��
?��HA�C3�=�:=q=u?��HA�C2��                                    By��  �          @XQ��<(����
?�z�A�C4W
�<(�<��
?�z�A�C3�f                                    By�0  �          @X���@  >#�
?ǮA�G�C0��@  >B�\?ǮAڸRC0J=                                    By��  �          @Vff�>{>�z�?�G�A��C.z��>{>��
?�G�A�(�C-޸                                    By�|  �          @\(��C�
>.{?ǮA�p�C0�{�C�
>L��?�ffA���C08R                                    By�"  �          @\(��C�
>L��?�ffAծC0.�C�
>u?��A���C/�{                                    By��  �          @q��U>.{?�  A�\)C1��U>W
=?�  A��HC0u�                                    By�n  �          @p���U>\)?�p�A��C1���U>.{?�p�Aڣ�C1                                    By�  T          @n�R�S�
=�?��HAڸRC1�H�S�
>#�
?��HA�Q�C1@                                     By��  �          @o\)�S�
=�?޸RAݮC1�q�S�
>��?޸RA�G�C1\)                                    By�	`  �          @n{�R�\=u?޸RA�G�C2���R�\=���?޸RA�
=C2W
                                    By�  T          @k��O\)=L��?�p�A�{C3��O\)=�Q�?�(�A��C2z�                                    By�&�  �          @hQ��O\)=���?�33A�p�C2J=�O\)>�?��A��C1�                                    By�5R  �          @j=q�P��=�G�?�33A�z�C2
=�P��>��?�33A�(�C1p�                                    By�C�  �          @h���P  >#�
?�\)AӅC10��P  >B�\?�{A�
=C0�
                                    By�R�  �          @e�L��>#�
?У�Aי�C1#��L��>B�\?�\)A��C0��                                    By�aD  �          @dz��K�>L��?���AՅC0xR�K�>k�?���A��HC/޸                                    By�o�  �          @fff�QG�>�Q�?���A���C-���QG�>Ǯ?���A�C-�                                    By�~�  �          @q��^{?   ?��A���C+�^{?�?���A�\)C+J=                                    By��6              ��O���O���O���O���O���O���O���O���O���O���O�                                  By���  	U          @������\?��?��RA���C%s3���\?��?�(�A�z�C%)                                   By���  �          @��R���
?s33?�A�  C&�R���
?z�H?��A�C&�{                                    By��(  �          @�G����R?z�H?�33A�33C&�����R?�  ?���A���C&�{                                    By���  T          @����  ?aG�?�z�A�=qC(Q���  ?h��?��A�=qC'�                                    By��t  �          @�=q��  ?^�R?�Q�A�p�C(^���  ?fff?�
=A�p�C'�R                                    By��  �          @�33����?fff?�Q�A��
C(
=����?n{?�A��
C'��                                    By���  �          @��\��Q�?n{?�A�{C'����Q�?u?�33A��C'=q                                    By�f  �          @�(�����?s33?��A�=qC&�\����?z�H?��A�  C&n                                    By�  �          @�(����\?��
?���A�C%�����\?��?�
=Az�RC%}q                                    By��  �          @�G��\)?��?��Av�RC%��\)?��?�\)Aq��C$�                                     By�.X  �          @��\�\)?��?�(�A��RC$\�\)?�?�Q�A�  C#�R                                    By�<�  �          @�33�s33?�  ?�A�
=C%J=�s33?��\?�33A��\C$�                                    By�K�  �          @xQ��fff?\(�?�Q�A���C&�{�fff?aG�?�A�Q�C&33                                    By�ZJ  �          @�ff�xQ�?���?�33A
=C$!H�xQ�?���?���Ay��C#�=                                    By�h�  �          @���qG�?�=q?��Al��C$��qG�?���?�G�Ag�C#�R                                    By�w�  T          @���~�R?���?c�
A>ffC �3�~�R?��?\(�A7�
C ��                                    By��<  �          @�{��33?�=q?G�A�C {��33?˅?@  A
=C��                                    By���  �          @�p�����?�  ?(�@�\)C�\����?�G�?�@�Q�C��                                    By���  �          @�ff���H?�33?��@��
C����H?�z�?�\@���C޸                                    By��.  �          @���fff?�>aG�@Dz�C�3�fff?�>8Q�@   C��                                    By���  �          @���p��?�
=>�=q@j�HC�\�p��?�Q�>k�@FffC�                                     By��z  �          @��H�tz�@�\>��R@�z�C�f�tz�@33>�=q@b�\C��                                    By��   T          @��xQ�@ff>��
@�C�f�xQ�@ff>�\)@e�C��                                    By���  �          @�  �n{@ ��>��@�
=C�f�n{@G�>�Q�@��
C��                                    By��l  �          @����p  @�
>�@\CJ=�p  @z�>��@�
=C.                                   By�
  �          @��\�u?�
=?�@�ffC\)�u?�Q�>�@�(�C8R                                   By��  
�          @�=q�w
=?�{?�@�\C8R�w
=?��?�@��C�                                    By�'^  T          @�  ��G�?�>�ff@�G�C�=��G�?�
=>��@�\)Ck�                                    By�6  �          @�p����R?�p�>�p�@�\)CǮ���R?��R>��
@{�C��                                    By�D�  
�          @��R��  ?��H>�G�@��C8R��  ?�(�>���@��C)                                    By�SP  
�          @�G����\?���>�p�@��
Cz����\?��H>��
@���Cc�                                    By�a�  "          @���xQ�?���>�z�@x��Cٚ�xQ�?�=q>�  @U�C�                                    By�p�  "          @Z�H�E?��H=��
?��C���E?��H=#�
?8Q�C��                                    By�B  �          @e��N{?���=�Q�?�p�C��N{?���=u?fffC�                                    By���  
�          @}p��a�?��>��@�C��a�?�ff=�G�?���C�                                    By���  �          @\)�c�
?�ff=�G�?�33CE�c�
?�ff=�\)?��C=q                                    By��4  �          @����g
=?�=q=��
?�33C��g
=?�=q=#�
?
=qC�                                    By���  T          @\)�c33?�<�>\C��c33?����
�k�C�                                    By�Ȁ  "          @\)�e�?�G�=�\)?p��CǮ�e�?�G�<��
>�{C                                    By��&  �          @n{�XQ�?Ǯ>\)@
=qC:��XQ�?Ǯ=���?�=qC0�                                    By���  �          @k��Tz�?���=���?У�C���Tz�?���=�\)?��C�f                                    By��r  
�          @j�H�S33?���=�G�?�33C0��S33?���=�\)?��C(�                                    By�  �          @n{�U�?��=��
?�Q�C� �U�?��=#�
?�C��                                    By��  T          @xQ��^{?�p�=�\)?���Cp��^{?޸R<�>�Ck�                                    By� d  �          @n{�XQ�?���=u?h��C��XQ�?���<��
>��RC\                                    By�/
  "          @p  �X��?���=u?fffC���X��?���<��
>�z�C��                                    By�=�  T          @j�H�R�\?�{=�\)?��
C�f�R�\?�{<�>ǮC��                                    By�LV  "          @u��[�?��H=�Q�?��C}q�[�?��H=L��?@  CxR                                    By�Z�  "          @r�\�Z=q?�z�=�Q�?��C��Z=q?�z�=#�
?.{C
=                                    By�i�  �          @n{�W
=?�{=#�
?.{Cn�W
=?�{    =L��Ck�                                    By�xH  
�          @s33�Z=q?�z�<��
>�Q�C)�Z=q?�zἣ�
��\)C)                                    By���  
�          @p���W�?�<#�
>B�\C�\�W�?������C��                                    By���  �          @o\)�W
=?��    ��\)C��W
=?У׽L�Ϳ:�HC{                                    By��:  T          @n�R�Vff?�논��ǮC��Vff?�녽�\)��ffC�3                                    By���  T          @n{�Vff?��ͽ�\)���Cp��Vff?��ͽ�G��ٙ�CxR                                    By���  "          @p  �Y��?�=q���Ϳ˅C��Y��?�=q����\)C\                                    By��,  �          @n{�Y��?\���
��33C��Y��?\�u�z�HC�3                                    By���  �          @i���U?��R    =L��C�U?�p��#�
�z�C                                    By��x  _          @^�R�I��?�(����
=qC���I��?�(���\)����C��                                    By��  "          @`  �E?У׾.{�-p�C5��E?�\)�W
=�\(�CJ=                                    By�
�  
�          @`���E�?��L���Q�CxR�E�?���  ��G�C�\                                    By�j  �          @`���Dz�?�������\)Cn�Dz�?�zᾳ33���C�\                                    By�(  "          @^�R�C33?�33�u�{�C���C33?�녾�\)��{C�R                                    By�6�  �          @`���B�\?޸R�B�\�I��C&f�B�\?޸R�u�}p�C=q                                    By�E\  �          @dz��Fff?޸R�aG��`  C�H�Fff?޸R��=q����C�R                                    By�T  �          @j�H�HQ�?�׾�33���C��HQ�?�\)�������HC{                                    By�b�  �          @n{�G
=?�(����H���
C���G
=?�������  C��                                    By�qN  �          @���Z�H@
=�
=q��\Cc��Z�H@����33C�
                                    By��  "          @tz��S33?�\)������=qCh��S33?�{������C�{                                    By���  �          @p���QG�?�=q��{��{C���QG�?��þǮ����C�H                                    