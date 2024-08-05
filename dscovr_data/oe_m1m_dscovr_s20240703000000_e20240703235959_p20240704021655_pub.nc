CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240703000000_e20240703235959_p20240704021655_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-04T02:16:55.213Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-03T00:00:00.000Z   time_coverage_end         2024-07-03T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy^A@  T          @o\)�\@J�H���\���HB�G��\@P�׿��\�|��B�
=                                    By^O�  
�          @h�ÿ��@<�Ϳ������B�����@Dz΅{���B�
=                                    By^^�  �          @a녿�Q�@!�����\B�=��Q�@,(�����Q�B���                                    By^m2  �          @h�ÿ�ff@*�H�ٙ���z�B����ff@2�\���R����B���                                    By^{�  
�          @j�H��\@#33�����HC ���\@*�H��(�����B�                                    By^�~  )          @k��G�@p��\�ÅC���G�@$zῪ=q��33Ck�                                    By^�$            @n�R�ff@��Ǯ��  C��ff@"�\��\)��ffC�                                    By^��  
(          @p  �{@Q��������C�{@ �׿�z�����Cs3                                    By^�p  T          @p���G�@ff��{��RC�G�@�R����33Cn                                    By^�  T          @qG���H@ff��������C
=��H@\)�����C	+�                                    By^Ӽ  
�          @q���R?����z���
C����R@ff��z���C��                                    By^�b  T          @u��&ff?�����
=C
=�&ff@�\��Q���33C�f                                    By^�  "          @u��0  ?��ÿ���p�C�=�0  ?��H���
���C��                                    By^��  T          @w��$z�@ff��(����C� �$z�@\)����\)C
��                                    By_T  
Z          @x���-p�?�Q��   ��G�CaH�-p�@������HCn                                    By_�  "          @{��8Q�?�녿�33��p�C���8Q�@G���G���=qC�H                                    By_+�  T          @y���/\)?������33Cp��/\)@녿����p�Cp�                                    By_:F  T          @x���-p�?����ff�C��-p�?�(����H��{C��                                    By_H�  �          @y���%@����C�H�%@���Q���\C�f                                    By_W�  "          @vff�z�@�����33C��z�@ff���H���HC��                                    By_f8  
Z          @x����H@������  C���H@=q��{����C33                                    By_t�  "          @|���+�@{��\)��Q�CO\�+�@ff���H�̸RC
�3                                    By_��  "          @|���H��?��H��z���C}q�H��?��ÿ�����C�H                                    By_�*  
�          @y���,(�@
�H��ff��(�C��,(�@33�����
=C��                                    By_��  
Z          @�  �@��?������
��p�C)�@��@z�����33C}q                                    By_�v  �          @����B�\?�p��G��z�C{�B�\?���
=q� 
=C�R                                    By_�  T          @����=q@Tz�@2�\B�
B�.��=q@G
=@AG�B0{B�Ǯ                                    By_��  "          @��
�#�
@S�
@7
=B#�B�p��#�
@E@EB3�RB�u�                                    By_�h  
�          @��R��Q�@_\)@.�RB��B�녾�Q�@R�\@>{B'��B���                                    By_�  �          @�\)�8Q�@q�@33A�
=BɊ=�8Q�@g
=@#�
BG�Bʊ=                                    By_��  T          @�
=���@p  @ffA�B�����@e�@&ffB��B���                                    By`Z  "          @�\)�c�
@qG�@\)A�{Bή�c�
@g
=@   B�B��)                                    By`   	�          @���\@qG�?��Aƣ�B��\@h��@��A��B�p�                                    By`$�  T          @�Q��\)@n�R?�Q�A�B�\��\)@e�@(�A�\B�3                                    By`3L  
�          @�\)��{@p��?�A��\B�\)��{@hQ�@�
A�G�B��)                                    By`A�  
�          @�\)�˅@q�?�ffA�G�B���˅@i��@33A�  B��                                    By`P�  
S          @�(�����@l��?�z�A�G�B��ÿ���@dz�@
=qA�ffB�aH                                    By`_>  T          @��Ϳ�(�@n{?�(�A��B�aH��(�@e�@{A�33BٸR                                    By`m�  �          @����\@j=q@��A�\)B�W
���\@`��@Q�B(�B��)                                    By`|�  T          @��R��
=@dz�@  A���B߮��
=@Z=q@�RB�B�                                     By`�0  
�          @�
=���@n�R@A�z�Bڨ����@e�@�A��B��                                    By`��  "          @����H@\��@
=B ��B�  ���H@Q�@%�B��B�                                    By`�|  �          @�(����@e@�A��B�B����@\��@�
A�
=B��H                                    By`�"  	�          @����\)@XQ�@
=B�RB�8R��\)@N{@%�B�B�8R                                    By`��  
�          @�{���
@p  ?��A���BҊ=���
@i��?��
AɮB�ff                                    By`�n  
�          @�����\@hQ�?�\AȸRB�G����\@`��@   A�33B�L�                                    By`�  T          @�z�u@e�?�A�=qB��u@\��@�A��B���                                    By`�  �          @���}p�@r�\?���A�p�B�33�}p�@l��?�\)A��B��                                    Bya `  �          @�  �z�H@q�?�33A�\)B�\�z�H@j�H?��A�p�B��                                    Bya  T          @�  �s33@Z=q@�B33B���s33@P��@"�\B{BԀ                                     Bya�  
�          @�\)��@aG�@�\A�(�B��H��@XQ�@  B �B�=q                                    Bya,R  T          @��R���H@Y��@
�HA�ffB�����H@P  @Q�B	��Bܨ�                                    Bya:�  )          @�ff���@P  @G�B(�B�Ǯ���@G
=@p�B=qB��                                    ByaI�  
�          @����
@U�@(�A�ffB�{���
@L(�@��BG�B�                                    ByaXD  �          @�p���Q�@Z�H?��A�ffB���Q�@S33@ffA�{B�33                                    Byaf�  T          @��׿�p�@^�R?���A��\B�
=��p�@X��?��
A�{B�{                                    Byau�  �          @������@Q�@
=A�
=B؞����@I��@33B
��B�\                                    Bya�6  �          @���k�@X��@Q�A��B�G��k�@P��@�B
{B�u�                                    Bya��  �          @z�H��33@Q�?�(�A���B���33@J=q@
=qB��B�z�                                    Bya��  
�          @u�\)@O\)@�
B��B�녾\)@G
=@  B\)B��                                    Bya�(  �          @r�\�u@I��@�
B��B�=q�u@AG�@\)B=qB�W
                                    Bya��  "          @s33���@Fff@
�HBp�B��׾��@>{@BB�\                                    Bya�t  [          @o\)>.{@1G�@�RB'  B��f>.{@(Q�@(��B4=qB��                                     Bya�  
�          @s�
�#�
@8��@{B"�B���#�
@/\)@(Q�B/G�B�Ǯ                                    Bya��  
�          @o\)>���@/\)@ ��B(��B�=q>���@%@*=qB5�B�u�                                    Bya�f  
�          @p��?�R@7�@�
B�
B�z�?�R@.�R@{B%�B�L�                                    Byb  T          @l��=�G�@5@ffB��B��==�G�@,��@ ��B+z�B�Q�                                    Byb�  T          @l��>�z�@0��@(�B%��B�{>�z�@'�@%B2\)B�p�                                    Byb%X  �          @l(�>�p�@5�@z�BffB���>�p�@,��@{B(��B�8R                                    Byb3�  "          @k�>��@9��@  Bp�B��{>��@1G�@��B#  B��                                    BybB�  T          @k�>���@:=q@\)BB�8R>���@1�@��B"(�B��3                                    BybQJ  
�          @qG�>�(�@1�@!G�B'ffB�Q�>�(�@(��@*=qB3��B�ff                                    Byb_�  �          @o\)>L��@E�@Bp�B���>L��@=p�@  B�RB�W
                                    Bybn�  �          @u?   @Q�@<(�BI�\B�B�?   @{@C�
BUp�B���                                    Byb}<  
�          @l��>��@A�@ffB
�B�\)>��@:�H@��B�\B���                                    Byb��  T          @k���  @7�@�\B(�B��)��  @/\)@(�B&�B�L�                                    Byb��  T          @l(����@,(�@   B*��B��H���@#�
@(Q�B6B�p�                                    Byb�.  
�          @l�;�(�@.�R@p�B&�B�녾�(�@&ff@&ffB2(�B���                                    Byb��            @mp���@=p�@
=qB��B����@6ff@�
Bz�B��                                    Byb�z  "          @qG���  @Y��?���A��B�� ��  @Tz�?�\A�=qB��3                                    Byb�   T          @p�׽u@Tz�?�Q�A��
B��f�u@N�R?���A�RB���                                    Byb��  
�          @o\)��z�@J�H?�z�A�{B�����z�@Dz�@z�BQ�B��                                    Byb�l  "          @qG�?Tz�@*�H@!G�B(G�B��=?Tz�@"�\@)��B3
=B��                                    Byc  �          @p  ?��@!�@,��B8�\B���?��@��@4z�BCz�B�G�                                    Byc�  �          @n�R=�Q�@,(�@   B+z�B�
==�Q�@$z�@(Q�B6�B��
                                    Byc^  �          @mp�����@<��@��B  B�B�����@6ff@�B�
B���                                    Byc-  
�          @j�H?.{@*�H@�HB%z�B��=?.{@#33@#33B/��B�G�                                    Byc;�  
�          @hQ�?(��@,(�@B ��B��=?(��@%�@p�B+\)B�aH                                    BycJP  
�          @g
=<#�
@!�@"�\B4p�B��<#�
@=q@)��B?
=B��                                    BycX�  "          @c�
>���@(��@�B$�B���>���@!�@��B.�B��)                                    Bycg�  "          @^{?��@3�
?�(�A��B�\)?��@.{?���BG�B�B�                                    BycvB  
�          @]p�?xQ�@2�\?�A�z�B���?xQ�@,��?�Q�B��B��=                                    Byc��  
�          @_\)?Q�@0  ?��RB
=B��f?Q�@*=q@
=BB��
                                    Byc��  "          @e�>#�
@�R@$z�B7B��{>#�
@
=@*�HBA��B�8R                                    Byc�4  
�          @c�
>��@'�@��B)��B�k�>��@ ��@ ��B3�\B�#�                                    Byc��  
�          @^{=�Q�@"�\@ffB+
=B��=�Q�@(�@p�B4�HB�Ǯ                                    Byc��  
�          @]p���@3�
@   B��B���@.{@�B\)B��                                    Byc�&  "          @^{?J=q@'�@�
B��B�k�?J=q@!�@
=qB�
B�Q�                                    Byc��  �          @[�?W
=@0  ?�\)B(�B���?W
=@*�H?�p�B=qB��
                                    Byc�r  [          @W�?�=q@,(�?�(�A�(�B�Q�?�=q@'
=?�=qB��B�8R                                    Byc�            @W�?�p�@   ?��B��Bp�?�p�@�H?��RB{B|z�                                    Byd�  �          @Vff?p��@-p�?ٙ�A�
=B��?p��@(��?�B(�B���                                    Bydd  �          @Vff?��\@5?�z�A�  B�ff?��\@1�?\A�
=B���                                    Byd&
  "          @Vff?��
@7
=?���A�B�u�?��
@333?��RAԏ\B��R                                    Byd4�  "          @Y��?z�@5�?��A�(�B��)?z�@0��?��B�B�G�                                    BydCV  �          @XQ�?Q�@3�
?�Q�A�B�z�?Q�@/\)?��B {B��q                                    BydQ�  T          @[�?��@1G�?���A�ffB}ff?��@-p�?�ffA׮B{�                                    Byd`�  T          @Y��?\@#�
?�{A�33Bm�?\@   ?ٙ�A�Bk{                                    BydoH  �          @W�>�@4z�?���A�B��
>�@0��?ٙ�A�{B�p�                                    Byd}�  �          @X��?+�@#�
@BG�B�?+�@�R@�B!33B��f                                    Byd��  T          @Z=q?.{@!�@Q�B�\B��?.{@��@{B$\)B�Ǯ                                    Byd�:  �          @Z�H?
=q@C33?�
=A�  B��?
=q@@  ?��
AծB���                                    Byd��  "          @[�>�33@;�?�Q�A��B�G�>�33@7�?��A�=qB���                                    Byd��  
(          @Z�H>\)@2�\?��HBG�B�W
>\)@.{@33B
=B�33                                    Byd�,  "          @X��>�33@:=q?�Q�A��
B�ff>�33@6ff?��A���B��                                    Byd��  	�          @X��>��
@Dz�?��A�p�B�33>��
@A�?�Q�A�Q�B�
=                                    Byd�x  T          @P  ��Q�@Fff?@  AZ�RB�k���Q�@Dz�?Y��Ax(�B�p�                                    Byd�  T          @U<�@P��=L��?uB�{<�@P��>#�
@1G�B�{                                    Bye�  �          @c33��33@<��?0��A?33B�q��33@;�?G�AX��B�{                                    Byej  
�          @l��?G�@!�@{B+B��?G�@p�@"�\B2p�B��                                    Bye  �          @k�>��@8��@��B�RB��>��@4z�@B��B���                                    Bye-�  �          @l�;#�
@5�@ffB\)B�k��#�
@0��@�B%�B���                                    Bye<\  
�          @dz�+�@2�\@
=B{B�(��+�@.�R@(�B�\B�                                    ByeK  �          @[�����@.{?��A/�B�����@,��?+�AF�\B�W
                                    ByeY�  T          @]p���\)@6ff>W
=@`  B�k���\)@5>�z�@��\B��=                                    ByehN  "          @j�H��@*�H?У�A��HB�aH��@'�?ٙ�A���B�Q�                                    Byev�  �          @k���G�@��@�BQ�B�����G�@�@  B=qB���                                    Bye��  
�          @hQ���
?�@+�BF��CO\���
?���@.{BJQ�C�R                                    Bye�@  �          @e���p�?�
=@7
=BW�HB����p�?�{@:=qB\��B��{                                    Bye��  T          @_\)�  =u@(Q�BEz�C2�
�  �#�
@(Q�BE�C4E                                    Bye��  �          @\����\?�p�@{B8z�C�)��\?�@   B;Q�C!H                                    Bye�2  T          @S�
��p�@�@{B,ffB�=��p�?�(�@G�B1{B��f                                    Bye��  "          @G���G�@
�H?�G�A���B�\��G�@��?ǮB p�B��H                                    Bye�~  
�          @Fff��  @G�@
=B,ffB��H��  ?�(�@
=qB1{B�                                    Bye�$  "          @E�O\)@33?��B=qB��)�O\)@��?�Q�B{Bۊ=                                    Bye��  
�          @2�\��G�@�?W
=A�Q�B���G�@ff?aG�A�ffB�k�                                    Byf	p  �          @DzῈ��@�?�=qBG�B�8R����@��?��B��B�\                                    Byf  T          @DzῘQ�@�?�  B ��B�\��Q�@33?�ffB�B�L�                                    Byf&�  �          @O\)?:�H@
�H@  B1B�\?:�H@Q�@�\B6�B�W
                                    Byf5b  
�          @O\)?c�
@Q�@  B1
=B���?c�
@@�\B5�B�Ǯ                                    ByfD  �          @O\)@�?�=q?�Q�B��B�@�?��?�(�B�\B�                                    ByfR�  
(          @Tz�?�?�\)@   Bz�B7�\?�?�=q@�\Bz�B5G�                                    ByfaT  �          @S�
@G�?�(�?��Bp�B!=q@G�?�
=?�z�B
=B                                      Byfo�  �          @Tz�?��@ ��@�B��BK��?��?�p�@�
B  BI�                                    Byf~�  
�          @U?�  @{@
=B�Br�R?�  @�@��B!��Bq�                                    Byf�F  
�          @W
=?�{@�@�\B,��B|
=?�{@��@�B0
=Bzp�                                    Byf��  �          @R�\?��
@�@��B!��B�p�?��
@  @
�HB%(�B���                                    Byf��  
�          @P  ?n{@z�@z�BG�B�33?n{@�\@ffB!ffB��                                    Byf�8  
�          @O\)?}p�@Q�@�RB.
=B�W
?}p�@ff@��B0��B��                                    Byf��  "          @S�
?��@�R@p�B'z�B��H?��@��@\)B*G�B�G�                                    Byfք  	�          @S33?��@G�@
=B�RB}p�?��@  @��B!\)B|Q�                                    Byf�*  "          @O\)?\(�@  @
=qB'ffB��?\(�@{@(�B*
=B���                                    Byf��  �          @Mp�?xQ�@Q�@�B+�B��f?xQ�@ff@p�B.\)B�aH                                    Bygv  \          @K�?z�H@�@(�B.B���?z�H@33@{B1
=B�#�                                    Byg  
�          @L��?&ff@{@p�B/
=B�p�?&ff@(�@\)B1\)B��                                    Byg�  �          @Mp�?&ff@��@�RB0��B�B�?&ff@�@  B2��B��                                    Byg.h  T          @Mp�>L��@33@��B.Q�B�  >L��@�@{B0p�B��f                                    Byg=  T          @K�>��@�@�BG�B�ff>��@ ��@(�BI�HB�L�                                    BygK�  
�          @H�þ���@�R@	��B.�B�.����@p�@
�HB0B�G�                                    BygZZ  �          @G���{@
=?�B  B�=q��{@ff?�
=B�B�W
                                    Bygi   T          @Fff?G�?��@B6�B�B�?G�?�\)@ffB8�B��                                    Bygw�  
�          @E���@!G�?�\)B�BƔ{���@ ��?У�B�Bƨ�                                    Byg�L  �          @A녿:�H@1G�?\(�A�
=B�k��:�H@1G�?^�RA��B�u�                                    Byg��  T          @C�
��?�
=?�\)B
=B�p���?�?��B��B��3                                    Byg��  �          @Dz�Ǯ?���@z�BEp�Bʙ��Ǯ?�Q�@z�BFp�BʸR                                    Byg�>  �          @Dz�>aG�?�=q@,��B~Q�B�aH>aG�?���@-p�B(�B�8R                                    Byg��  T          @Fff<#�
?�ff@%Bl�B�u�<#�
?��@%Bm=qB�u�                                    Bygϊ  �          @Dzῂ�\@��?�(�B=qB�W
���\@��?�p�BB�p�                                    Byg�0  �          @E�����@	��?��
A�ffB�B�����@	��?��
A��B�W
                                    Byg��  �          @E���ff@{?�B��B����ff@{?�B
=B�3                                    Byg�|  "          @N�R�aG�?s33@Dz�B��qB��)�aG�?s33@Dz�B���B��                                    Byh
"  �          @X�ý�?�33@J=qB�ǮB����?�33@J=qB�ǮB���                                    Byh�  T          @W��Y��?�{@%BN�HB��ÿY��?�{@%BNB��                                    Byh'n  "          @Z�H����?�@p�B8z�B�ff����?�
=@p�B8=qB�Q�                                    Byh6  �          @]p���{?�@Q�B.
=C���{?�@�B-�RC�3                                    ByhD�  �          @^{���@Q�@�B33B�\���@��@�BB��                                    ByhS`  �          @^{�\@��@Q�B=qB�\�\@��@�B��B��f                                    Byhb  �          @^�R��\)@�H@z�BB�녿�\)@�H@�
B  B�                                    Byhp�  �          @X�ÿ�  @?�
=B
��B�Q��  @ff?�B
�B��                                    ByhR  �          @XQ쿺�H@��?�\)B�RB򞸿��H@=q?�{B�RB�k�                                    Byh��  �          @X�ÿ�Q�@�?��A�=qB��׿�Q�@(�?У�A�{B�k�                                    Byh��  �          @Y����Q�@��?˅A�
=C�q��Q�@G�?�=qA��HC�)                                    Byh�D  �          @W��\)@Q�?(��A6�RC^��\)@��?&ffA2�\CO\                                    Byh��  "          @Z�H�1G�?�
=��Q����C�1G�?��������C��                                    ByhȐ  "          @^{�#33?
=�
=q�{C&��#33?��
=q��C'L�                                    Byh�6  �          @`�����>�p��{�6C+:����>�33��R�7{C+�q                                    Byh��  �          @k���þ��<���V  C@#���þ��H�<���U�C@�)                                    Byh�  �          @i����þ�Q��;��U�HC=z���þǮ�;��Uz�C>@                                     Byi(  
           @j=q�33���?\)�[G�C@�3�33�   �>�R�Z�RCA�\                                    Byi�  �          @j=q��ÿB�\�7��NQ�CG����ÿJ=q�7
=�MffCHO\                                    Byi t  "          @l���p���G��:=q�Q  C?Y��p����:=q�PffC@:�                                    Byi/  �          @j�H���R=����Dz��c��C1  ���R=u�Dz��c�C2!H                                    Byi=�  
Z          @n�R�
�H��\)�@���W�C;W
�
�H���
�@���Wz�C<c�                                    ByiLf  
Z          @o\)����W
=�7
=�G��C8�������  �7
=�GQ�C9ٚ                                    Byi[  �          @mp��{�W
=�0  �?�C8�)�{��  �0  �?\)C9�=                                    Byii�  �          @j�H��u�3�
�H\)C9����z��3�
�G��C:��                                    ByixX  �          @l(��  �#�
�:�H�Q�C4G��  �u�:�H�Q��C5z�                                    Byi��  �          @mp��(�=����1��C(�C1�=�(�=u�2�\�CG�C2��                                    Byi��  �          @l(��
=>���.�R�BC(�{�
=>�
=�/\)�Cz�C)�3                                    Byi�J  �          @l�����>��H�1��C
=C(����>�(��2�\�C�
C)�                                    Byi��  �          @l�����?���-p��<33C&(����?���.{�=33C'J=                                    Byi��  �          @mp��!�?+��'��3�HC%!H�!�?�R�(Q��4��C&5�                                    Byi�<  T          @j=q�(�?333�5�K�C"33�(�?&ff�7
=�MG�C#��                                    Byi��  
�          @j=q���?5�5��J�HC"5����?&ff�5�LG�C#�)                                    Byi�  �          @g���\?E��8���S�C@ ��\?5�9���U��C �=                                    Byi�.  �          @j�H�\)>.{�7
=�OffC/���\)=�G��7
=�O�C1:�                                    Byj
�  �          @i���p��L���8���R
=C5:��p���G��8Q��Q�HC6�                                    Byjz  �          @h���33<��333�J�C3E�33�#�
�333�J�C4��                                    Byj(   T          @h���z�#�
�2�\�H�C7���z�k��2�\�H=qC9�=                                    Byj6�  
�          @j=q�G���Q��7
=�N{C68R�G��#�
�6ff�MC8                                      ByjEl  �          @i���ff�#�
�1��GffC4@ �ff���
�1��GG�C5��                                    ByjT  �          @g������
�/\)�EffC;�{���Ǯ�/\)�Dz�C=��                                    Byjb�  "          @h���p��@  �#�
�3�\CD�)�p��O\)�"�\�1�CFO\                                    Byjq^  
�          @g
=�p�>W
=�,(��I�
C.u��p�>\)�,(��JQ�C0^�                                    Byj�  
�          @c�
��
=>Ǯ�:�H�_��C(����
=>����;��a�C+
=                                    Byj��  
�          @b�\���W
=�5��T(�C9Ǯ�������4z��S\)C;�3                                    Byj�P  �          @h�����>�33�;��UC*Ǯ���>���<(��V�RC-�                                    Byj��  �          @l����
>���B�\�\�RC'
��
>\�C33�^�C)�=                                    Byj��  "          @l����
>�(��A��]�C(E��
>����C33�^ffC*��                                    Byj�B  �          @k��G�>�
=�B�\�_�C(E�G�>��
�C33�`ffC*�f                                    Byj��  �          @k��
�H?�R�:=q�P��C#�3�
�H?��;��R�\C&E                                    Byj�  �          @j�H�Q�?!G��:�H�RC#���Q�?��<(��TC&�                                    Byj�4  �          @k���ff?B�\�Mp��v�RC�R��ff?&ff�N�R�y�CJ=                                    Byk�  "          @l�Ϳ�p�?h���O\)�wQ�Ch���p�?L���QG��{=qC��                                    Byk�  
�          @r�\�3�
?�\�{�#=qC)���3�
>�(���R�$�\C+ff                                    Byk!&  �          @qG��&ff?���)���3(�C'  �&ff?�\�*�H�4�HC)                                      Byk/�  "          @qG��$z�?
=�+��5�C'�$z�?   �,���7=qC)�                                    Byk>r  "          @q��8Q�?^�R����C#0��8Q�?G���
��HC$                                    BykM  "          @p  �5�?\(��33�33C#)�5�?E������C$�                                     Byk[�  �          @o\)�B�\?B�\��\�Q�C%�q�B�\?.{�z��\)C'ff                                    Bykjd  "          @o\)�>�R?
=���  C(Ǯ�>�R?   �p����C*aH                                    Byky
  "          @n�R�8Q�?���33�z�C)5��8Q�>��z��{C*�q                                    Byk��  �          @n�R�2�\?\)���� \)C(�=�2�\>���=q�"
=C*z�                                    Byk�V  T          @o\)�/\)?@  ��H�!C$�q�/\)?&ff����$�C&�R                                    Byk��  �          @mp��&ff?k�����&p�C ��&ff?Q��\)�)�C"��                                    Byk��  �          @n{�0��?�z����33C(��0��?����{��HC�
                                    Byk�H  "          @n{�8Q�?��
�z���\C k��8Q�?n{���C"
=                                    Byk��  T          @n{�<��?��׿�Q���C�q�<��?�ff��p��\)C xR                                    Bykߔ  T          @n�R�:�H?��׿��H� 33C�)�:�H?��� �����C c�                                    Byk�:  T          @l���9��?xQ��G����C!���9��?^�R��
���C#B�                                    Byk��  �          @j=q�<(�?�=q��\)���\C�=�<(�?}p������C!O\                                    Byl�  "          @dz��'
=?Y���
=q���C"��'
=?@  ����C$\                                    Byl,  �          @fff�/\)?�33�����RCٚ�/\)?����33� 33CY�                                    Byl(�  �          @g��5�?�������HC���5�?�=q��Q���C(�                                    Byl7x  "          @dz��-p�?��\��
=�G�C�H�-p�?�
=���R��HC��                                    BylF  �          @e�G
=?�����\���\C�q�G
=?��
�����p�C��                                    BylT�  �          @i�����?�G��ff�(��CǮ���?���=q�.{C&f                                    Bylcj  �          @hQ��;�?�ff�˅�ң�C#��;�?��H��33���
C�                                     Bylr  "          @j=q�z�?s33�'��8�HC���z�?Q��*=q�=�C �{                                    Byl��  �          @fff��R?����
=�$  C���R?z�H��H�(�C�                                    Byl�\  "          @i���=q?�  �!��0=qCn�=q?^�R�%��4��C =q                                    Byl�  �          @hQ���?����\)�.G�C�{��?���#33�3�C�=                                    Byl��  �          @j�H�p�?����{��HC���p�?�Q��33�ffC��                                    Byl�N  �          @mp���H?�����H�$  CE��H?��R�\)�*{C�                                    Byl��  �          @o\)�/\)?�=q�����p�CY��/\)?޸R���R��p�C�=                                    Bylؚ  �          @p  �9��@G����
����C.�9��?��H�����Q�C�                                    Byl�@  �          @p���@��?�ff��(���\)C��@��?�p��������RC
                                    Byl��  �          @o\)�G�?�Q��G��£�C33�G�?��Ϳ�=q���
C �f                                    Bym�  �          @l(��#33?Tz��\)�+��C"
�#33?.{�"�\�/p�C%&f                                    Bym2  �          @l(��6ff?�G���=q����CO\�6ff?���
=���C�                                    Bym!�  �          @i����@
=��������C�q��@G��˅��Q�C!H                                    Bym0~  �          @g���  ?Q��L(��wQ�CxR��  ?�R�N�R�}\)C��                                    Bym?$  
�          @mp���=q>�G��Tz��~��C$xR��=q>k��UǮC+�
                                    BymM�  �          @n{���R>��W�W
C"�q���R>u�X����C*�                                    Bym\p  �          @mp���(�>�p��W��\C%𤿼(�>���X����C.                                      Bymk  �          @o\)����>��\��33C!T{����>k��^{  C*(�                                    Bymy�  T          @n{���R>��
�^�Rz�C%�{���R=�Q��_\)�3C/��                                    Bym�b  �          @mp�����?B�\�S�
�~�C�����?
=q�VffQ�C�                                    Bym�  �          @l(����?333�\(���C0����>���^�R33C}q                                    Bym��  �          @k���z�?c�
�P���{p�C���z�?+��S�
k�C��                                    Bym�T  �          @n{��G�?W
=�X���C=q��G�?(��\(�ǮC#�                                    Bym��  "          @o\)��{?s33�U�~(�C{��{?8Q��Y��(�C)                                    BymѠ  �          @qG���G�?xQ��Z=q�)CY���G�?=p��^{G�C�q                                    Bym�F  �          @q녿��?��\�XQ��~p�C�R���?��
�]p�C��                                    Bym��  �          @s33��G�?�ff�Z�H��C ���G�?�ff�`  �RC�                                   Bym��  �          @u���(�?�33�\(���C�q��(�?fff�aG�33C��                                   Byn8  
(          @s�
����?�=q�^�R�=C=q����?Tz��b�\�
C��                                    Byn�  �          @s33��=q?���]p��Cc׿�=q?c�
�a���C�\                                    Byn)�  "          @s33��ff?�{�^{��Czῆff?Y���c33W
C                                    Byn8*  
�          @u�����?:�H�c33�
C�쿙��>���e��C�                                    BynF�  �          @u�:�H?���aG���B��f�:�H?���g
==qB��{                                    BynUv  T          @w
=����>�{�i��8RC#Q쿐��=u�j�H�RC0޸                                    Bynd  �          @vff��G�>�33�l(��=C ��G�=�\)�mp�Q�C/�                                    Bynr�  
�          @u����>�=q�g�k�C'(������#�
�hQ�ffC4B�                                    Byn�h  �          @u���
=>��fff�Cٚ��
=>L���hQ�ffC*^�                                    Byn�  �          @tzΉ�?B�\�e�G�C:Ή�>���hQ��Cz�                                    Byn��  �          @w���{?O\)�g
=G�C���{?��j=q�C�
                                    Byn�Z  �          @x�ÿ�G�?���g�ǮCp���G�>��R�j=q�HC&{                                    Byn�   
�          @w����?���e��C�����>���g
=��C)�                                    Bynʦ  �          @u���Q�>���fffǮCuÿ�Q�>8Q��hQ�33C+�\                                    Byn�L  T          @vff��(�>�p��hQ�.C#쿜(�=�\)�h���
C0��                                    Byn��  "          @vff��p�>�z��hQ��C&�=��p��#�
�h���RC4�
                                    Byn��  �          @tz῱�>#�
�c33
=C-���녾\)�c33�C9��                                    Byo>  
�          @u��=q=�Q��fffp�C00���=q�W
=�e  C=0�                                    Byo�  T          @vff�����
�j=q��C4�ÿ������h��=qCC��                                    Byo"�  �          @vff��Q�>u�hQ��C(ff��Q콏\)�h����C7E                                    Byo10  �          @vff����>���hQ���C,������#�
�g���C;�
                                    Byo?�  �          @u�z�H���R�k�CE��z�H�!G��h��u�CTǮ                                    ByoN|  �          @u��z�h���a�CZ!H��z῜(��\(����CbaH                                    Byo]"  
�          @w��u�G��h���CZ�q�u�����c�
u�Cd�=                                    Byok�  
�          @y���0�׿z�H�l��#�Cj��0�׿�ff�fffffCr!H                                    Byozn  
�          @y���u��Q��l���3C�c׾u�\�e��\C��{                                    Byo�  T          @z�H��׿�z��`����\C~J=��׿�(��Vff�kffC�W
                                    Byo��  
b          @|(��L�Ϳ�p��`  �~�C�:�L����\�U��i��C�XR                                    Byo�`  
�          @u��}p���=q�I���mz�Cm�f�}p���{�?\)�[(�Cq�q                                    Byo�  �          @\)�W
=?h�ÿ����HC$ٚ�W
=?:�H�   ����C'Ǯ                                    Byoì  T          @�Q��U�?�G�������G�C=q�U�?�=q���H�陚C!�R                                    Byo�R  T          @�Q��W�?��׿�=q��p�C!u��W�?s33��
=��=qC$5�                                    Byo��  T          @����`��?�\�J=q�5�CL��`��?�
=�s33�[�Cc�                                    Byo�  �          @�Q��aG�?��!G��Q�CǮ�aG�?޸R�L���7�
C��                                    Byo�D  �          @|���Z�H?����\)�ffCJ=�Z�H?�
=�������\C�{                                    Byp�  
�          @~{�h��?}p���ff�x��C$�=�h��?aG������p�C&^�                                    Byp�  T          @|���j�H?}p�����s�
C$���j�H?aG���������C&��                                    Byp*6  �          @~{�n�R?J=q�\(��L  C(\�n�R?333�p���]C)^�                                    Byp8�  �          @~�R�n�R?�\)�B�\�1C#L��n�R?���^�R�K
=C$s3                                    BypG�  T          @~�R�s�
?(�ÿp���Z{C*(��s�
?\)��  �h��C+�{                                    BypV(  T          @�Q��s�
?8Q쿀  �g�C)G��s�
?�R�����w�C*��                                   Bypd�  �          @����o\)?W
=��Q���z�C'O\�o\)?8Q쿢�\��  C)&f                                   Bypst  T          @�G��q�?aG�����p  C&�f�q�?E���\)��  C(��                                    Byp�  �          @�Q��o\)?�G��}p��f�\C$�\�o\)?h�ÿ���~{C&\)                                    Byp��  "          @����u�>�����|Q�C,���u�>�p�������p�C.��                                    Byp�f  T          @�Q��r�\�L�Ϳ�����
C4���r�\�B�\���\��Q�C6�
                                    Byp�  T          @����tz�u��
=��33C7���tzᾸQ쿓33��p�C9aH                                    Byp��  �          @����fff��\)����ә�C5!H�fff�����\��\)C8!H                                    Byp�X  �          @��H�p�׾�\)���
��z�C80��p�׾�G����R���
C:��                                    Byp��  �          @����x�þW
=��=q�x  C7��x�þ��
��ff�qG�C8��                                    Byp�  
�          @�33�w��+�����xQ�C=ٚ�w��J=q��G��f{C?z�                                    Byp�J  �          @�z��i���n{�ٙ����HCBE�i����{�˅���CD��                                    Byq�  "          @���a녿�Q�ٙ����CF���a녿�\)�Ǯ����CI:�                                    Byq�  T          @��
�^{����������HCJ�R�^{��\)��(���ffCM�                                    Byq#<  T          @��H�XQ쿾�R��G��ˮCK�{�XQ��
=��=q��{CNh�                                    Byq1�  �          @���k��xQ쿽p���=qCB�{�k����׿�{��(�CE�                                    Byq@�  �          @���o\)�.{��G����HC>J=�o\)�W
=������C@��                                    ByqO.  "          @�33�|(��#�
��G��f�HC4���|(��#�
��  �d(�C6L�                                    Byq]�  �          @��H����>�{�+���\C/)����>�=q�333�ffC08R                                    Byqlz  �          @�33��  >����Q��9C/����  >W
=�Y���@z�C1
=                                    Byq{   �          @��H�z=q>#�
��33��p�C1��z=q<��
������C3��                                    Byq��  �          @���|(�<��
��33��(�C3�R�|(���G�������C5��                                    Byq�l  �          @���~�R>��R�u�W�C/��~�R>L�Ϳ}p��^=qC1&f                                    Byq�  
�          @��
�|(��u��z�����C7z��|(���p������~�HC9n                                    Byq��  �          @�(��y�����R�����p�C8���y����׿����{C:��                                    Byq�^  "          @�z��w����Ϳ�����C5��w���=q��33��ffC7�q                                    Byq�  "          @����w
=�B�\��  ��Q�C6Ǯ�w
=��p���(���ffC9h�                                    Byq�  �          @��w������R���
C;  �w��&ff��
=���
C=��                                    Byq�P  �          @���s33��  ��\)���
C7�q�s33��G���������C:�H                                    Byq��  
�          @�ff�tz�=u��p�����C3+��tz�����(���=qC6Q�                                    Byr�  
�          @���}p��(�ÿ������C=s3�}p��O\)�������\C?�
                                    ByrB  �          @�\)�u��ff��z����CCW
�u��(����\���CE��                                    Byr*�  �          @�
=�vff��{���\���CD&f�vff��G������x��CF+�                                    Byr9�  �          @�ff�o\)����  ��Q�C<�o\)�=p�������C?.                                    ByrH4  �          @�\)�u�
=q��
=��
=C;�q�u�=p���������C>��                                    ByrV�  �          @��R�u��
=��{���RC<� �u��J=q���
��z�C?�)                                    Byre�  �          @�
=�|�;�ff������
=C:�=�|�Ϳ�R�����33C<�R                                    Byrt&  �          @�����  ��R������HC<�)��  �J=q��  ���RC?(�                                    Byr��  �          @���~�R��Ϳ����G�C;���~�R�5��G���(�C>#�                                    Byr�r  �          @�Q��~�R�B�\��=q��33C>�
�~�R�n{��(�����CA�                                    Byr�  T          @�  �����&ff��\)�t  C=������J=q���\�_33C?�                                    Byr��  "          @����Q�O\)��=q�k�C?n��Q�p�׿u�R{CA:�                                    Byr�d  
�          @�\)�z�H��ff��z��~�RCC��z�H��Q쿁G��]��CD�                                    Byr�
  �          @�\)�|�Ϳc�
������{C@���|�Ϳ�������k�CB��                                    Byrڰ  �          @�ff�w
=�z��ff���C<z��w
=�E����H���RC?Y�                                    Byr�V  "          @��z=q<��
��(�����C3���z=q�.{���H����C6�=                                    Byr��  �          @�p��w�=��ͿǮ���
C2���w���G��Ǯ��C5��                                    Bys�  �          @��u�>8Q�����p�C1T{�u��#�
��33��z�C4��                                    BysH  �          @�(��qG�?
=�Ǯ��p�C++��qG�>�p���\)��G�C.aH                                    Bys#�  �          @����n{��Q���H�ŮC5c��n{���
��Q���=qC8�                                    Bys2�  �          @���o\)��׿޸R���
C;��o\)�333��z���=qC>�{                                    BysA:  "          @��q녾�녿ٙ���
=C:@ �q녿#�
�У���Q�C=�f                                    BysO�  �          @���w��u�\���C4ٚ�w���=q��  ��
=C7�                                    Bys^�  �          @���y��>.{��33��p�C1xR�y�����
��z�����C4Q�                                    Bysm,  �          @��
�w�=�G���33���HC2n�w���Q쿳33����C5Q�                                    Bys{�  �          @���tz�?Q녿�33���C'��tz�?(���  ���C*��                                    Bys�x  �          @�p��u?G���33���HC(��u?z῾�R��{C+u�                                    Bys�  T          @�p��tz�?��ÿ�p���(�C$^��tz�?c�
��\)��(�C&�R                                    Bys��  �          @�z��o\)?B�\��{����C(xR�o\)?��ٙ����C+�R                                    Bys�j  �          @���o\)?O\)�������C'� �o\)?z�ٙ����C+:�                                    Bys�  T          @�p��n{?��Ϳ�����\C ��n{?�zῼ(�����C"�                                     BysӶ  �          @�
=�qG�?��\��33��
=C!aH�qG�?���������\C$Y�                                    Bys�\  �          @��R�r�\?�zῴz����HC"�3�r�\?s33�Ǯ����C%�R                                    Bys�  T          @��qG�?�33�����G�C#��qG�?p�׿�����C&                                    Bys��  �          @�p��r�\?�\)������C#���r�\?h�ÿ��R��G�C&�                                    BytN  �          @�
=�q�?�p���z���z�C!�3�q�?�G���=q���C%�                                    Byt�  T          @�
=�o\)?���������HC ���o\)?�=q�У����C#�
                                    Byt+�  �          @��R�o\)?��\��(����
C!8R�o\)?����33���C$u�                                    Byt:@  �          @�ff�n{?��������C .�n{?�\)��������C#L�                                    BytH�  "          @�{�l(�?�Q쿮{��33C���l(�?�(��Ǯ��z�C!��                                    BytW�  
�          @��R�dz�?ٙ������z�C��dz�?�p���\)��z�C�\                                    Bytf2  �          @��qG�?L�Ϳ˅��G�C(��qG�?�Ϳ�Q����C+��                                    Bytt�  T          @���s33?O\)��(����C'��s33?z������(�C+O\                                    Byt�~  T          @�  �dz��p��\��33CMٚ�dz��Q쿞�R��G�CP��                                    Byt�$  �          @����c33���H�У����
CM�3�c33��Q쿬����CP�)                                    Byt��  �          @�G��`�׿�  �ٙ���33CNn�`�׿�p���z���(�CQxR                                    Byt�p  
�          @���\(��G��������CRc��\(��\)���\���CU�                                    Byt�  
Z          @����u���  ������G�CB��u����R��33��\)CE�)                                    Byt̼  �          @�  �u��:�H�����C>�
�u��z�H�����{CBff                                    Byt�b  �          @�\)�o\)��33����{C9W
�o\)�#�
���Σ�C=                                    Byt�  �          @���q녽�Q�ٙ����C5ff�q녾�Q��z����HC9aH                                    Byt��  T          @�p��w�=��Ϳ�ff��{C2}q�w��\)�����C6�                                    ByuT  �          @��\�r�\>��Ϳ������RC.��r�\>#�
���R��  C1�=                                    Byu�  �          @��H�s33?z`\)����C+G��s33>�Q쿸Q�����C.��                                    Byu$�  �          @�(��tz�>#�
��ff��
=C1�f�tzὸQ�Ǯ���C5aH                                    Byu3F  �          @��R�{�>��H�������C,��{�>��������C08R                                    ByuA�  �          @�
=�c�
?���  ��{C���c�
?�\)�\��=qC��                                    ByuP�  
�          @�\)�k�?�z῝p���33C���k�?�Q쿽p����C��                                    Byu_8  
R          @�\)�j�H?����\���C�{�j�H?�Q�\����C�{                                    Byum�  
�          @�Q��r�\?�녿�=q��C�f�r�\?�33������C#�                                    Byu|�  �          @�\)�s�
?�33��Q���
=C�s�
?�Q쿴z���G�C"�H                                    Byu�*  "          @���r�\?��H���R���C���r�\?�p����H���RC!�                                    Byu��  �          @��R�j�H?�
=���H��p�Cff�j�H?��H��(���33CQ�                                    Byu�v  �          @�\)�p  ?��
��(����
C� �p  ?����������C ��                                    Byu�  �          @�{�mp�?��R������33C��mp�?��\��
=��C!\                                    Byu��  "          @�{��  ?G��n{�N�RC(����  ?�R��ff�iG�C+33                                    Byu�h  �          @�����>��
�Q��6�HC/�����>8Q�\(��@(�C1�                                    Byu�  �          @����
>L�Ϳ5��C1G����
=��
�:�H� ��C2��                                    Byu�  
�          @�������>Ǯ�J=q�0��C.xR����>�  �Y���<��C0k�                                    Byv Z  �          @�33�\)>��H�@  �)�C-��\)>�33�Q��9p�C.��                                    Byv   �          @���\)?
=q�O\)�6�\C,Y��\)>Ǯ�c�
�H��C.ff                                    Byv�  T          @�������?G���������C))����?333�����C*0�                                    Byv,L  �          @�p���=q?Q녾��R��G�C(�{��=q?B�\��ff���C)u�                                    Byv:�  �          @�p���=q?B�\�������C)h���=q?.{�����
C*�                                    ByvI�  �          @�����\?333��Q���Q�C*=q���\?!G����ӅC+=q                                    ByvX>  �          @�ff���
?:�H��{��p�C)�����
?(�þ��ʏ\C*�)                                    Byvf�  
�          @����\?O\)������HC(� ���\?:�H�
=q��{C)޸                                    Byvu�  �          @�����?Q녾�Q���{C(�=���?@  �   ��33C)��                                    Byv�0  �          @�p����?\(���33���
C(����?J=q�   ���
C)
=                                    Byv��  �          @�p����H?=p���Q����C)�����H?+�����(�C*�3                                    Byv�|  �          @�p����H?L��<��
>��RC(�R���H?J=q����\)C)
                                    Byv�"  �          @������?Tz�=u?O\)C(n���?TzὸQ쿝p�C(xR                                    Byv��  �          @�(����?=p��\)� ��C)�����?333��=q�s33C*&f                                    Byv�n  T          @������?�R�z�� (�C+J=���?�\�.{��RC,ٚ                                    Byv�  �          @�=q�}p�?L�;�G���p�C(���}p�?5���p�C)�H                                    Byv�  �          @���{�?L�Ϳ����C(u��{�?0�׿+���
C)�R                                    Byv�`  �          @���|��?�=q�\)���RC$���|��?��������p�C%E                                    Byw  �          @�=q�|��?h�þ�z�����C&���|��?Y����ff���C'�                                    Byw�  T          @��\�|(�?���8Q��!�C%=q�|(�?}p���Q���(�C%�                                    Byw%R  �          @�(��|(�?��׾�����C#���|(�?��
�+���\C%h�                                    Byw3�  
�          @�(��x��?�
=�8Q�� ��C#��x��?��
�k��N�RC%&f                                    BywB�  
�          @���w
=?��H�=p��%G�C"���w
=?�ff�p���T��C$��                                    BywQD  
�          @�33�o\)?��ÿ�G��ep�C �\�o\)?�{���R���HC#p�                                    Byw_�  �          @����s33?�z�W
=�;�
C���s33?�p�����s\)C")                                    Bywn�  	�          @��R�z=q?�{�:�H��C �\�z=q?����xQ��T��C"�f                                    Byw}6  �          @�  ��  ?�=q����  C!����  ?��H�@  �#�C#(�                                    Byw��  �          @����G�?��R�k��C�
C"���G�?�
=��ff��(�C#                                    Byw��  �          @�{�z�H?�\)���陚C �R�z�H?�  �G��+33C"T{                                    Byw�(  
Z          @��{�?�{���
��ffC �3�{�?��\�����
C"�                                    Byw��  �          @��R�\)?�p���(���p�C"�H�\)?��׿(����C$=q                                    Byw�t  �          @�ff�}p�?�  �(����C"z��}p�?�{�Tz��6�\C$J=                                    Byw�  �          @�{�~�R?��H��
=��G�C#��~�R?�{�#�
�G�C$h�                                    Byw��  
�          @��xQ�?��
�����n=qC%��xQ�?O\)���R��G�C(0�                                    Byw�f  �          @���y��?��\��  �^=qC%ff�y��?O\)�����C(G�                                    Byx  
�          @���xQ�?n{�����}G�C&�\�xQ�?333�������C)�{                                    Byx�  	�          @����s�
?aG�������33C&�3�s�
?(��\��{C*��                                    ByxX  �          @���qG�?aG����
���C&Ǯ�qG�?z��
=���C+Q�                                    Byx,�  
�          @�{���\?E�����33C)E���\?(�ÿ(���(�C*�\                                    Byx;�  �          @�����?B�\�aG��B{C)J=����?z῁G��_33C+�{                                    ByxJJ  "          @��\)?8Q쿁G��`��C)���\)?������|  C,��                                    ByxX�  T          @�ff�|(�?L�Ϳ�(���\)C(u��|(�?�Ϳ������\C,                                    Byxg�  �          @�p��~{?5��ff�jffC)��~{>��H�����\C,�q                                    Byxv<  T          @�
=��Q�?=p�����i�C)����Q�?���Q�����C,�
                                    Byx��  "          @��R�x��?�
=�n{�O
=C#!H�x��?xQ쿑��33C%�                                    Byx��  T          @�
=�w�?�33�aG��AG�C )�w�?�������z�HC"                                    Byx�.  
�          @���w�?��R�W
=�8  C��w�?����\)�uG�C!��                                    Byx��  �          @�Q��w
=?���h���Dz�CB��w
=?�=q��Q���  C �R                                    Byx�z  �          @�\)�z=q?�G��xQ��U��C"#��z=q?���������\C%�                                    Byx�   �          @�\)�z=q?z�H��Q�����C%ٚ�z=q?:�H��{��33C)k�                                    Byx��  T          @�Q��}p�?��H�\(��;\)C"���}p�?��\��=q�m�C%�
                                    Byx�l  �          @�  �~{?���=p�� Q�C!��~{?�녿}p��V�RC#�R                                    Byx�  T          @�  �\)?�ff�!G��	�C!��\)?�33�aG��?\)C#�                                    Byy�  �          @�  �|��?��
�Q��3
=C"��|��?������hz�C$�=                                    Byy^  T          @�\)�}p�?�ff�333���C!�=�}p�?��׿s33�O�C$                                      Byy&  �          @�
=�y��?��Ϳ\(��=G�C �{�y��?�33��\)�vffC#�                                    Byy4�  
�          @�\)�|(�?��
�L���/
=C"��|(�?������d��C$�                                    ByyCP  
�          @���u?��ͿG��(��CaH�u?�zῊ=q�mG�C�{                                    ByyQ�  �          @���{�?��H�   ���HC�H�{�?�=q�J=q�,(�C!W
                                    Byy`�  �          @�  �z�H?�\)�c�
�@��C ��z�H?�zΐ33�z�\C#��                                    ByyoB  "          @�  �x��?�  �Q��2{C�f�x��?�ff��{�r{C!}q                                    Byy}�  �          @����}p�?�p��&ff��C}q�}p�?��ÿp���J�\C!��                                    Byy��  �          @����{�?����R���C���{�?��׿k��F�HC ��                                    Byy�4  �          @�Q��xQ�?�
=��
=��p�C���xQ�?Ǯ�B�\�#�
C#�                                    Byy��  �          @�  �|��?�G��\���C�|��?�33�0�����C p�                                    Byy��  "          @�\)����?��\����a�C"z�����?�Q������C#�=                                    Byy�&  T          @����=q?��.{��RC#����=q?�{������
=C$�                                     Byy��  �          @����  ?��׾.{�G�C ���  ?��þ�ff�ÅC!�                                    Byy�r  �          @�  �\)?��W
=�1�C }q�\)?���   �ָRC!n                                    Byy�  �          @�\)����?�G���z��\)C"������?������  C#��                                    Byz�  �          @�
=����?�Q�\���C#}q����?����R�\)C$�H                                    Byzd  �          @�\)��Q�?����z���33C#Q���Q�?�ff�Q��2=qC%E                                    Byz
  
�          @�\)��  ?�
=�&ff�=qC#�{��  ?��\�c�
�AC%�                                     Byz-�  "          @�  �~�R?�p��(�����C"޸�~�R?���h���F�HC%
                                    Byz<V  T          @�\)�}p�?�(��L���,��C�f�}p�?�녿   �ۅC ��                                    ByzJ�  
�          @�������?�=q��\��{C!������?�Q�G��)��C#�                                    ByzY�  T          @���z=q?�=q���
��=qC��z=q?�(��&ff���CQ�                                    ByzhH  �          @�\)�|��?�33�u�R{C#�\�|��?k���
=���C&�                                    Byzv�  "          @�=q���?u�J=q�(��C&޸���?E��xQ��Qp�C)ff                                    Byz��  T          @�=q���?p�׿k��Dz�C'+����?8Q쿌���k�C*{                                    Byz�:  "          @�=q���?���J=q�)��C%�����?Y����  �U�C(c�                                    Byz��  
�          @����33?�z������C$.��33?�G��Tz��2�RC&5�                                    Byz��  "          @��\���\?���(��ffC"n���\?��׿aG��;33C$�                                    Byz�,  
�          @��\���H?��R�!G��p�C#33���H?��ÿaG��;�C%T{                                    Byz��  "          @��H��=q?�(��W
=�333C 0���=q?�녿���{C!33                                    Byz�x  �          @�=q��G�?�G����Ϳ���C� ��G�?�����(����RC :�                                    Byz�  �          @�33����?�녽u�W
=C�\����?˅��
=����Cu�                                    Byz��  "          @����=q?�{=�Q�?��RC!����=q?��;W
=�/\)C!��                                    By{	j  �          @�ff��33?�=q�\)��C����33?�G�����Q�C��                                    By{  	�          @�p���=q?��?��@У�C 5���=q?�p�>8Q�@��CO\                                    By{&�  T          @�p����@G�>8Q�@�
C�H���@ �׾���:�HC�R                                    By{5\  �          @�\)��?�\)>W
=@z�C!����?У׾����z�C!��                                    By{D  
Z          @����(�?�ff>W
=@33C%����(�?�����
�Y��C%�f                                    By{R�  �          @��\��G�?�Q�>u@'
=C!z���G�?ٙ��\)��(�C!\)                                    By{aN  "          @���(�?�>��?�\)C!{��(�?�z�k��!G�C!+�                                    By{o�  �          @�G���z�?��\?
=q@��C(����z�?�{>���@P��C'��                                    By{~�  
�          @�����Q�?}p�?�R@�=qC)Y���Q�?���>\@�=qC(+�                                    By{�@  �          @���33?Tz�>�Q�@p��C+!H��33?c�
>#�
?ٙ�C*�                                    By{��  "          @�
=��p�?
=�L�Ϳ
=qC-� ��p�?녾8Q��C.                                    By{��  �          @�Q���ff?(�þL���p�C,Ǯ��ff?����{�p��C-s3                                    By{�2  �          @�ff��(�?W
==#�
>�(�C*����(�?Q녾����33C*޸                                    By{��  �          @������?u�����RC)L����?n{�u�,(�C)��                                    By{�~  �          @����Q�?z�H���
�W
=C(����Q�?s33�u�*�HC)G�                                    By{�$  �          @����?���=�?�C'p���?��;���C'p�                                    By{��  �          @����\)?aG�<�>���C)�R��\)?\(��.{���HC*(�                                    By|p  
�          @��\����?8Q콏\)�L��C+޸����?.{�k��'
=C,B�                                    By|  
�          @����
=?k�=�G�?�Q�C)����
=?k����Ϳ�Q�C)��                                    By|�  T          @�����ff?h��>�  @5C)����ff?p��=#�
>�G�C)E                                    By|.b  �          @��H��\)?z�H>�p�@�\)C(�{��\)?��>\)?���C(+�                                    By|=  �          @�G���{?xQ�=#�
>�C(�H��{?s33�8Q���\C)\                                    By|K�  "          @�p����?}p�=#�
>�(�C(T{���?z�H�B�\�{C(��                                    By|ZT  �          @�(���Q�?�G�<��
>�z�C'����Q�?}p��W
=���C(:�                                    By|h�  T          @��
��\)?��ͼ��
�W
=C&���\)?��þ�=q�J�HC'O\                                    By|w�  T          @����\?xQ�>.{?�p�C(�����\?}p��u�(��C(xR                                    By|�F  "          @�
=���
?h��>L��@�\C)p����
?n{���
�8Q�C)33                                    By|��  *          @�
=��z�?\(�=�Q�?��C*��z�?\(���G���  C*
=                                    By|��  ~          @����p�?B�\�u�0��C+=q��p�?8Q�k��+�C+�H                                    By|�8  
�          @��R���?333��Q쿆ffC+�����?(�þ�  �8Q�C,\)                                    By|��  
�          @�  ��p�?J=q�#�
�#�
C*޸��p�?E��L�����C+&f                                    By|τ  �          @�  ��(�?��\=�Q�?��\C(&f��(�?��\����޸RC(:�                                    By|�*  
(          @�\)����?���>k�@,��C#�)����?�33��Q쿈��C#��                                    By|��  �          @��\��\)?s33>.{?��HC)5���\)?u�L�Ϳz�C)�                                    By|�v  �          @�(���=q?0��=���?��C,=q��=q?333��\)�E�C,33                                    By}
  �          @�33��Q�?\(��B�\�C*=q��Q�?J=q�\����C*�q                                    By}�  �          @������?녿   ��z�C-������>�G���R��  C/                                      By}'h  
�          @�����{>�녿@  �	�C/G���{>aG��Q��
=C1n                                    By}6  T          @����?W
=�=p��=qC*O\��?&ff�h���&�RC,�=                                    By}D�  �          @����ff?W
=��R��Q�C*W
��ff?+��L���G�C,@                                     By}SZ  �          @�����?h�ÿ=p��=qC)����?5�n{�)C+Ǯ                                    By}b   T          @������?n{�+�����C)L����?=p��^�R�
=C+c�                                    By}p�  �          @�G���z�?s33�G��ffC)
=��z�?=p��z�H�3�C+p�                                    By}L  �          @�����z�?fff�@  �	p�C)�)��z�?0�׿p���,��C+�                                    By}��  T          @�Q����?�ff�c�
�#\)C'�����?O\)��{�MG�C*ff                                    By}��  �          @�����\?fff�Tz��{C)k����\?.{���\�=C,                                      By}�>  S          @�{����?^�R�W
=�G�C)������?&ff���
�?�
C,O\                                    By}��  �          @�ff��G�?c�
�W
=���C)s3��G�?+����
�@(�C,�                                    By}Ȋ  �          @��R����?\(��aG��#�C)�)����?�R����E��C,�
                                    By}�0  "          @�
=��G�?^�R�u�2{C)�f��G�?�R����TQ�C,�)                                    By}��  �          @������
?W
=�\(���C*+����
?(����
�>{C,��                                    By}�|  
�          @�  ��33?c�
�L���
=C)����33?+��}p��6�\C,�                                    By~"  �          @������?@  �0�����
C+W
���?\)�Y���\)C-z�                                    By~�  �          @�����
?L�Ϳ:�H��HC*�����
?���fff�&ffC-                                    By~ n  
�          @�Q���(�?:�H�Q��=qC+s3��(�?��xQ��2�\C-��                                    By~/  �          @�Q���z�?+��Y���(�C,5���z�>�G��}p��5p�C.��                                    By~=�  "          @�  ��p�?�\�0����\)C.���p�>��
�J=q��\C08R                                    By~L`  �          @�Q���\)>��������=qC0�R��\)>#�
��ff��p�C2+�                                    By~[  �          @��R��(�?
=�.{��(�C-���(�>�녿L�����C/8R                                    By~i�  "          @�{��(�>�(��\)��Q�C/���(�>�\)�#�
���C0��                                    By~xR  
�          @�ff��{���
��p����C4����{�.{��{�}p�C5�                                    By~��  
�          @�p������=q������C7&f������;��H��{C8��                                    By~��  T          @�{��=q���H�L���ffC9��=q�+��(����
=C;�H                                    By~�D  
�          @����R�aG����^�RC6�����R�   ��=q�Mp�C:�                                    By~��  �          @������R���þ�ff��ffC7�3���R��(���Q���=qC9(�                                    By~��  "          @�(����þ���p���33C9�����ÿ�;u�5�C:�
                                    By~�6  
�          @�����H��  �&ff��p�C6����H���Ϳ���
=C8��                                    By~��  �          @�
=���R���;�=q�G
=C50����R�.{�u�.{C5��                                    By~�  "          @�{��z�\)>\@��C:� ��z��ff?   @�=qC9=q                                    By~�(  T          @���=q�:�H?333A{C<����=q�
=q?\(�A   C:h�                                    By
�  T          @�{�����G�?J=qA  C=B������\)?uA3�C:�3                                    Byt  T          @��R��녿n{?5A
=C>���녿:�H?h��A)�C<�
                                    By(  �          @�ff��  �^�R?��AEC>\)��  �z�?��RAhQ�C:�R                                    By6�  
�          @���33��z�?�G�AnffCB���33�O\)?�G�A��C=�R                                    ByEf  T          @��������
=?�p�A��\CB�����5?�(�A��\C=(�                                    ByT  "          @����G����?�p�A�(�CB����G��(�@��A�
=C<)                                    Byb�  T          @�����G��
=q?�@�
=C:xR��G��Ǯ?.{AC8�)                                    ByqX  �          @�������xQ�?��\Aq��C?������!G�?�(�A�ffC;�q                                    By�  "          @�����G��(��?333A�C;�\��G���?W
=A�\C9�                                    By��  T          @����p��J=q?��
ADz�C=����p���\?���Ad��C:G�                                    By�J  T          @����33�s33?z�HA<��C?���33�.{?�Q�Ae�C<^�                                    By��  
�          @������\)?�A���CI+�����{@(�A֏\CB޸                                    By��  	�          @�(��|�Ϳ�33@��A�Q�CM�3�|�Ϳ��@#33A��
CFQ�                                    By�<  
�          @��������{@�
A���CL�)�������
@p�A��\CE��                                    By��  
Z          @�����ÿ���@ ��A�p�CLL����ÿ�G�@��A���CE^�                                    By�  "          @��\���׿޸R@G�A��CKT{���׿�
=@��A�\)CDJ=                                    By�.  
�          @�\)�u���G�@ffAә�CL���u���
=@{A��CE{                                    By��  
�          @�Q��mp�����@�
A�G�CN���mp���(�@,��B
�
CF#�                                    By�z  
(          @��\���ÿ�Q�@33Aȣ�CJ�R���ÿ�\)@��A���CC��                                    By�!   �          @�=q��z�˅?�33A��CH�q��zῇ�@�RA�
=CB^�                                    By�/�  �          @����z���?�z�A��CI�\��zῌ��@��Aޣ�CB�                                    By�>l  T          @�p���������@z�A�RCI+������n{@(Q�B��CA�                                    By�M  
(          @�z��~�R��(�@��A�  CKc��~�R����@'�Bz�CCn                                    By�[�  
�          @��H�z�H��(�@  Aݙ�CK�R�z�H����@'
=B��CC��                                    By�j^  �          @�(��vff��ff@��A�RCL���vff���@1G�B
(�CDh�                                    By�y  �          @���w���(�@�A��
CK���w����@.�RB�CCY�                                    By���  "          @����x�ÿ޸R@�HA��HCL��x�ÿ���@1�B
\)CCaH                                    By��P  �          @��}p����
@A�z�CL:��}p�����@-p�B�CC�R                                    By���  �          @�
=�^{�
=@4z�B(�CSY��^{��=q@P��B%=qCH��                                    By���  
�          @�ff�u���{@�RA�{CM�{�u���@7�B��CD�                                    By��B  �          @�ff�mp���Q�@'�A��
CO���mp���(�@AG�B��CF(�                                    By���  T          @�{�k���@!G�A���CQ���k���z�@>�RB=qCI                                      By�ߎ  T          @��R�l���
�H@�RA��HCR\)�l�Ϳ��H@<��BffCI��                                    By��4  
Z          @�ff�i���
=@$z�A��HCR�i�����@AG�B�HCH��                                    By���  �          @�
=�o\)���R@%�A��HCO���o\)���\@@  B  CF�3                                    By��  "          @�ff�u��z�@�HA��HCNu��u���R@4z�B�
CE��                                    By�&  �          @��{���  @��A�z�CL\�{����@0��BQ�CCxR                                    By�(�  �          @�����ÿ�
=@  A�{CJ�����ÿ��@&ffB �CB�3                                    By�7r  �          @���  ���
@G�A�G�CK����  ���@)��B��CC�                                    By�F  
�          @�{��G���p�@  A�33CK���G����@'�B �CC#�                                    By�T�  �          @��x�ÿ��@A㙚CM���x�ÿ�(�@/\)BCEp�                                    By�cd  T          @��H�g��Q�@��A�Q�CRz��g���Q�@7�Bp�CI�                                    By�r
  "          @��\�g���@��A�ffCQ�{�g���33@5B�
CI{                                    By���  �          @���}p���@�
Aȏ\CM���}p�����@�RA���CF}q                                    By��V  �          @��H��Q���?���A�Q�CM33��Q쿨��@�A�\CFE                                    By���  �          @�33���׿�33?�
=A��
CMJ=���׿��@ffA�ffCFs3                                    By���  �          @����G���{?�=qA���CL���G���=q@\)A�Q�CF(�                                    By��H  "          @��R�y����=q?�33A�z�CM��y�����
@�
A�\CF#�                                    By���  �          @�p���녿�33?У�A���CJ\��녿�?��RA�(�CD)                                    By�ؔ  "          @�{��녿�\?��
A��
CK����녿���?�
=A�z�CF
=                                    By��:  �          @�p��}p���\)?У�A�G�CMQ��}p����@33A��HCG\)                                    By���  
Z          @�p��x�ÿ���?��A�33CN�\�x�ÿ��H@A�33CH�=                                    By��  �          @��
�|(���33?�
=A��\CMǮ�|(���(�?�\)A�CHz�                                    By�,  �          @�(��{���(�?��HA��CN�f�{����
?�A��CIB�                                    By�!�  �          @�(��z=q��{?�\)A�{CMh��z=q����@�\A��CGh�                                    By�0x  �          @��
�|�Ϳ�33?�p�A�CJ���|�Ϳ�33@A�  CDE                                    By�?  �          @�(��z=q�У�?���A�33CJ�)�z=q���@p�A��CC��                                    By�M�  �          @��R�j�H��Q�@
=A��
CO�{�j�H��=q@"�\BG�CG�)                                    By�\j  �          @��\�]p���R@�RA�ffCT�\�]p���G�@>{BCK��                                    By�k  �          @����W
=�Q�@ffA��CWQ��W
=��Q�@8��B{CN��                                    By�y�  �          @����Z=q�"�\@�RA�G�CX�3�Z=q���@3�
BG�CPٚ                                    By��\  T          @����c�
�\)@�RA�p�CT@ �c�
�˅@.�RB=qCL
                                    By��  �          @����n�R��@�RA�  CNQ��n�R����@(Q�BCE�H                                    By���  �          @�=q�����ff?c�
A+\)C@������G�?���AZffC=�)                                    By��N  �          @�33���ÿ��R?�Q�Ac\)CCO\���ÿc�
?�(�A�\)C?)                                    By���  �          @����(���  ?��HA�(�CF���(�����?�ffA�=qCA                                    By�њ  �          @��H���
��{?ǮA��RCEE���
�h��?�{A�
=C?��                                    By��@  �          @��H��zΰ�
?���A�  CD33��z�Q�?���A�{C>��                                    By���  �          @�=q��zῇ�?�33A�Q�CA�\��z�
=?�\)A���C;��                                    By���  �          @��\��33���
?�z�A���CDu���33�O\)?�
=A���C>��                                    By�2  �          @�33�����G�?�p�A��\CC�f����Tz�?�  A�z�C>�f                                    By��  �          @�33���׿���?�\)A�\)CAW
���׿.{?���A�ffC<��                                    By�)~  �          @�33��G��Y��?�(�A��HC>����G���G�?��A�  C9�\                                    By�8$  �          @��\��녿W
=?�=qA\)C>c���녾�?�  A��RC9�=                                    By�F�  �          @������33?�
=A��CB�����=p�?�
=A�Q�C=n                                    By�Up  �          @��\��{���?���A~{CDh���{�k�?�{A��\C?�3                                    By�d  �          @�z����Ϳ�z�?c�
A(Q�CA�3���ͿaG�?�z�A\(�C>                                    By�r�  �          @����{��G�?��
AB=qC@!H��{�333?�G�AmC<�=                                    By��b  �          @���{�^�R?���Ab�\C>����{��?��A�33C:n                                    By��  �          @�z���녿s33?�p�A�\)C?� ��녿�?�
=A���C:��                                    By���  �          @�����׿��?\A���CA�����׿(��?�  A�(�C<O\                                    By��T  �          @�{���ÿ�=q?У�A���CK����ÿ��@33AĸRCE^�                                    By���  �          @�p���Q쿑�?��A�{CB8R��Q�333?��
A�z�C<�)                                    By�ʠ  �          @�z���ff��p�?�A�z�CFk���ff���?�  A��CAh�                                    By��F  �          @�(������ff?��A��CG��������?��A�=qCB
=                                    By���  �          @�z���=q���?�33A�z�CG�{��=q��ff@   A�\)CA��                                    By���  �          @����
=�У�?n{A0��CH���
=��=q?���A}G�CD�{                                    By�8  �          @��
��Q쿪=q?�p�Ai�CDk���Q�u?��
A��HC@�                                    By��  �          @����z�u?�G�An�HC?���z���?�(�A���C;Q�                                    By�"�  �          @�\)���Ϳ�
=?��
An�\CB33���ͿL��?��A���C=�=                                    By�1*  �          @�
=��(����
?��RAf�RCC� ��(��k�?��
A��C?0�                                    By�?�  �          @�\)��ff��G�?}p�A7�CB�R��ff�u?��
Ao�C?z�                                    By�Nv  T          @�G���=q�\?�{A}�CFu���=q��{?��HA�ffCA�R                                    By�]  T          @������׿�{?��
A�\)CG�����׿�z�?�33A���CBW
                                    By�k�  �          @���������p�?��AR=qCHǮ������\)?�ffA�ffCD�                                     By�zh  �          @�
=��{�"�\>�@���CS0���{�z�?���ATz�CP�q                                    By��  �          @��R�����  ?c�
A&�RCIQ�������H?��Ax��CF�                                    By���  �          @�\)�������?��Ap��CC������p��?˅A�\)C?}q                                    By��Z  �          @�ff���׿k�?��A�G�C?xR���׾�
=?��HA��C9J=                                    By��   �          @�
=���R��  ?˅A�\)CF�=���R���
?�
=A��\C@��                                    By�æ  �          @�����z�޸R?�
=A�z�CI����z῞�R@�AÙ�CCǮ                                    By��L  �          @�Q���33��G�?ٙ�A�p�CJ���33��  @ffA�\)CD�                                    By���  �          @��R����  ?�=qA�p�CC���:�H@A�  C=W
                                    By��  �          @�����Q��Q�?�\)AN�HCQ8R��Q�� ��?ٙ�A�33CM8R                                    By��>  T          @�G���G��
�H?�Q�A�33CN�H��G���(�?��HA��CI�H                                    By��  �          @�G���33��33?�p�A��HCD�H��33�u?��A��C?�=                                    By��  �          @�G���
=��@,(�B33C;33��
=>B�\@.�RB�C1p�                                    By�*0  �          @�G����ÿ�  ?��
A���CCn���ÿ=p�@�\A�C=O\                                    By�8�  
�          @�33��  ��G�?���A�(�CF����  �xQ�@�Aʏ\C@8R                                    By�G|  �          @��\���׿�p�?�G�A�=qCF#����׿xQ�@�A�(�C@�                                    By�V"  �          @��
������p�@�A��HCI�������\)@   A��CB��                                    By�d�  �          @��H��33�޸R?��RA�CI���33��z�@�A�(�CB�f                                    By�sn  �          @�33��{���H@�
A؏\CJ5���{��ff@+�A��\CB                                    By��  �          @�������\@p�A���CJ���������@%A�Q�CB�                                    By���  �          @�����ÿ��@  A�\)CI����ÿ�  @&ffA�33CA!H                                    By��`  �          @��
���ÿٙ�@(Q�A�ffCJ޸���ÿs33@>�RB(�CAL�                                    By��  �          @��\��(���G�@#�
A�(�CH(���(��J=q@7�B	ffC>�)                                    By���  �          @�����zῑ�@ffAƸRCB�{��z���@z�A�C;.                                    By��R  �          @�����G����H@�
Aۙ�CC�3��G��\)@#33A�{C;p�                                    By���  �          @�  ������?�A�{CEG����\(�@ffA��
C>�R                                    By��  �          @��R���׿\?��RA��CG�����׿s33@�
A�33C@}q                                    By��D  �          @�����G���z�?�
=A�ffCI)��G�����@33A�p�CBO\                                    By��  �          @����N�R��@G
=Bp�CU�R�N�R���@e�B7  CI                                    By��  �          @�������33?��AI�CD�=�������?�A�(�C@��                                    By�#6  �          @�  ���ÿ�p�?�=qA��CL�����ÿ�  @�\A���CGG�                                    By�1�  �          @����z�H�,(�?ǮA��CVs3�z�H�(�@p�Aљ�CQ+�                                    By�@�  �          @���l���6ff?�Q�A�\)CY�H�l����
@Q�A�RCT�                                    By�O(  �          @��R�w��+�?ǮA���CV���w���@p�A�p�CQff                                    By�]�  �          @�  �z=q�!�?�\A��RCT���z=q��p�@�A�33CN�)                                    By�lt  �          @�
=�u��1G�?�A�=qCW�)�u��33@ffA��CS                                      By�{  �          @���}p��7�?n{A+�
CW��}p��!�?��A�(�CT��                                    By���  �          @����G��!G�?�=qAz�\CS����G��?�Q�A�G�COT{                                    By��f  �          @��R�����#33?O\)A�CS�{�����  ?���A�z�CPz�                                    By��  �          @�
=��{��=q>��@�\)CF����{��33?^�RA!�CD��                                    By���  T          @��R��
=��Q�>�
=@��
CE  ��
=���
?G�A�CC5�                                    By��X  �          @�
=������?
=q@�  CF������\)?n{A,��CD\)                                    By���  �          @�\)���H��33?c�
A%�CG�����H��\)?��Ap��CD�{                                    By��  �          @���������?�R@���CC�
������?s33A/�CA�                                     By��J  �          @�\)��{���H?��AS\)CBn��{�^�R?�z�A��
C>��                                    By���  �          @�\)��
=��=q?G�A  CC����
=��=q?��AK�C@�H                                    By��  �          @����Q��G�?}p�A9�CIW
��Q쿹��?�A�
=CE                                    By�<  �          @�  �x���>�R?n{A+33CYxR�x���(��?�A�=qCV(�                                    By�*�  T          @�����Q��333?��AH��CV�H��Q���H?�\A�  CS
                                    By�9�  �          @��\�|���1G�?��RA��CW��|����\@
=qAʸRCR)                                    By�H.  �          @��\�������?�  A��
CO�����ÿ��@�\A�\)CJ��                                    By�V�  �          @��\��z��	��?��A~ffCN#���z��(�?�33A��
CIff                                    By�ez  �          @��
����	��?�33AP��CM�)������
?�A��RCI��                                    By�t   �          @��\��
=�z�?��Am��CL�\��
=��z�?��A�
=CH\)                                    By���  �          @������\��R?�33A��CP�H���\��z�@=qA噚CJ&f                                    By��l  �          @��
�fff�@2�\B=qCT���fff���@S33B ��CK)                                    By��  �          @�33�s�
�z�@)��A��CP���s�
����@EB�CG)                                    By���  �          @�=q�l����@$z�A��HCS���l�Ϳ��@E�B�CJ�f                                    By��^  �          @����U��"�\@,��BG�CY@ �U���  @P��B#��CO�q                                    By��  T          @�p��N{�>�R@.�RA���C^���N{��@Y��B$�HCV)                                    By�ڪ  T          @�z��N�R�QG�@33A��HCaaH�N�R�$z�@Dz�Bz�CZ��                                    By��P  �          @����J=q�_\)@�A���CcǮ�J=q�6ff@7
=B�\C]��                                    By���  �          @�z��Q��e?�33A��
Cc�H�Q��C33@!G�A���C^�                                    By��  �          @���_\)�Y��?ٙ�A��C`@ �_\)�6ff@ ��A�G�C[=q                                    By�B  �          @�33�Y���[�?�{A�Ca8R�Y���9��@(�A���C\p�                                    By�#�  �          @�(��c�
�L(�?�\)A�
=C]�{�c�
�&ff@(Q�A�(�CX&f                                    By�2�  �          @��\�p  �B�\?��Ax��C[
=�p  �%@�A��CV�                                    By�A4  �          @����}p���@   A�\)CR��}p����
@"�\A��CL:�                                    By�O�  �          @��H��G����?�Q�A�G�CR�3��G�����@\)A�Q�CLJ=                                    By�^�  �          @���y����@\)A���CP
=�y����\)@<(�Bz�CGaH                                    By�m&  �          @���~{�(��?�  A�  CU�f�~{�
�H@Q�Aə�CP��                                    By�{�  �          @������H��H?У�A�CR�
���H��@��A�p�CM&f                                    By��r  T          @����xQ��7�?��A�ffCX}q�xQ���H@z�A�33CS��                                    By��  
�          @�  �u��8Q�?���A�{CX�3�u��(�@z�AîCTs3                                    By���  �          @�  �q��/\)?�p�A��CW�q�q��p�@Q�A�  CRY�                                    By��d  �          @����s33�1G�?�A�ffCX��s33�  @z�A݅CR�H                                    By��
  �          @�=q�Vff�QG�?��A��RC`J=�Vff�,(�@)��A��CZ�3                                    By�Ӱ  �          @��R�|(��	��@$z�A�Q�CP�)�|(���
=@A�B�CG��                                    By��V  �          @�ff�~�R�G�@(Q�A�CN�R�~�R���@C33B�CE��                                    By���  �          @�33���\��@
�HAʸRCO����\���H@'�A��CG��                                    By���  �          @�33�w
=���R@)��A��COE�w
=��  @Dz�BQ�CE�q                                    By�H  �          @��H�u��\@(��A�Q�CP��u���@Dz�B��CF�=                                    By��  �          @�z������'�?˅A��HCU{�����Q�@��AΣ�CO�                                    By�+�  
�          @���xQ��
=q@�A噚CQ��xQ쿽p�@9��B�CH��                                    By�::  �          @��������
�H?�\)A���CN����׿У�@
=A�z�CHٚ                                    By�H�  T          @���z�H�ff@�A��CR���z�H�ٙ�@333B��CKh�                                    By�W�  �          @��������!G�?�33A�p�CS�
�������H@{A�{CMǮ                                    By�f,  �          @����(��	��@��A�{COz���(����
@*�HA���CHG�                                    By�t�  �          @�G����
��R@	��A�z�CS����
��{@,��A��HCLB�                                    By��x  �          @�G��|(��0��@p�AŅCW  �|(��
=@5�Bp�CP:�                                    By��  �          @���|���1�@��A�  CW)�|�����@5�B ��CPh�                                    By���  �          @��\�mp��W�?xQ�A/\)C^B��mp��AG�?��A��C[#�                                    By��j  �          @��\�p  �p  >�(�@��Ca��p  �aG�?�{Al��C_:�                                    By��  �          @��\�~�R�aG�>�G�@�C]u��~�R�S33?�ffAb{C[�H                                    By�̶  �          @�=q��\)�Mp�>.{?�=qCY+���\)�C�
?xQ�A'�CW��                                    By��\  �          @�����R�K�>��H@��CY!H���R�=p�?��\A]�CW&f                                    By��  �          @�����  �C33?@  ACW�H��  �0��?�p�A�Q�CU
=                                    By���  �          @����~�R�H��?�Q�A~{CZO\�~�R�+�@�A�\)CV�                                    By�N  �          @���X���~�R<�>��RCe���X���u?��\A3\)Cd��                                    By��  �          @�
=�n{�fff?��@ÅC`��n{�Vff?�
=A�C^�                                    By�$�  �          @��_\)�s33>�=q@?\)Ccu��_\)�g�?���AW�Cb                                    By�3@  �          @����]p��q�>�{@s33Cc���]p��e�?��\Adz�Ca�R                                    By�A�  �          @�p��q��^{?
=@��HC^���q��N{?�
=A��HC\aH                                    By�P�  �          @�{�j=q�hQ�?�\@�ffC`���j=q�X��?�33Az=qC^�
                                    By�_2  �          @��R�n{�^{?z�HA-��C_  �n{�HQ�?�ffA���C\                                      By�m�  �          @��Y���q�?O\)AQ�Cd��Y���^{?�(�A��Ca��                                    By�|~  �          @����1���G�>��@>{Cm��1����H?��Am�Ck�=                                    By��$  �          @��R�����׽��Ϳ���Cvk�������?�=qA>�HCu޸                                    By���  �          @�Q��{�����G���Cy@ ��{��G�?���AA�Cx                                    By��p  
�          @�ff�33���;#�
��  Cs��33����?}p�A/�Cs+�                                    By��  �          @�����H���׼#�
��Q�Cw�f���H��(�?�z�AP(�Cw�                                    By�ż  �          @�p���Q���G��#�
��G�Cw�f��Q�����?���AJ�RCwQ�                                    By��b  �          @�{��
=������z��J=qCx��
=��\)?fffA\)Cw�                                     By��  �          @���޸R�����ff��\)Czz�޸R���
?B�\Ap�CzW
                                    By��  �          @�����H���
�=p��33Cz�����H����>��@��
Cz��                                    By� T  �          @��R����
=�k��   Cv8R����z�?n{A&{Cu��                                    By��  T          @�G��:=q��33>aG�@!�Cj�{�:=q�z�H?��HA^�HCiaH                                    By��  �          @��ÿ������H��Q���
=C{)������G�?E�A�Cz��                                    By�,F  �          @�  ��
=��
=?���AU��CH����
=��{?�G�A�
=CD��                                    By�:�  �          @�=q���
��=q?��A��CJ����
���@
�HA�p�CE�                                    By�I�  �          @����z��  ?�=qA�\)CI����zῠ  @��A͙�CC�                                    By�X8  �          @�33��\)��Q�@�A��HC5!H��\)>��H@z�A��C-�\                                    By�f�  
�          @�G����>��R@!�A��C08R���?h��@��A�G�C(�H                                    By�u�  �          @����  >�{@"�\A�RC/���  ?p��@��A��
C(�\                                    By��*  �          @�\)����=�G�@Q�A��HC2�)����?.{@�A�\)C+��                                    By���  �          @�  ���
>#�
@{A�
=C2����
?@  @
=A�=qC*�\                                    By��v  �          @�ff��G�?   @{A��C-����G�?�=q@�A�{C&��                                    By��  �          @�����?\)@�HA���C-  ����?�\)@{A���C&#�                                    By���  �          @�Q����?!G�@!�A�z�C,#����?�(�@�
A�z�C%{                                    By��h  �          @�����z�>�Q�@'
=A�=qC/����z�?xQ�@��A�\)C(�                                    By��  �          @�
=���R?@  @)��A�(�C*�3���R?�{@��A�{C#��                                    By��  �          @�G����?��@*=qA�C'^����?�z�@�A�33C ��                                    By��Z  �          @�������?\(�@'�A�G�C).����?��H@A��
C"
                                    By�   �          @�����\)?�{@%A��C&#���\)?�Q�@��A��C\)                                    By��  �          @�  ���?z�H@(��A�z�C'n���?�=q@A��
C J=                                    By�%L  �          @�G���33?�@#�
A�C,�R��33?�z�@
=A�\)C%��                                    By�3�  �          @�  ���H=��
@!G�A�(�C2����H?.{@�A��C+��                                    By�B�  �          @������;W
=@"�\A�
=C6������>Ǯ@!G�A���C/33                                    By�Q>  �          @�  ���ÿ�R@#�
A�C;�
���ü#�
@(Q�A�
=C4
                                    By�_�  �          @�{��{��@(Q�A��C:�
��{=���@+�A�33C2��                                    By�n�  �          @�����Q��G�@,(�A���C5Y���Q�?�@(��A��C-T{                                    By�}0  �          @�z����?\(�@$z�A�\)C)�=���?�Q�@�\A�p�C"��                                    By���  �          @�
=��
=?�(�@{A��HC%����
=?�G�@�A�Q�C��                                    By��|  �          @�����ff?�ff@�Ȁ\C$����ff?�?�(�A��RC��                                    By��"  �          @�33����?�
=@�RA��
C�=����@	��?��
A�{C��                                    By���  T          @����?�G�@��AָRC!ٚ���@�\@�\A�{C&f                                    By��n  �          @�������@z�@A��HCs3����@#33?��A�(�C��                                    By��  T          @�z���  ?�ff@�HA�CE��  @�
?�Q�A�G�C
                                    By��  �          @���(�?�(�@   A�G�C#5���(�@ ��@ffA�\)C��                                    By��`  �          @������?�@��A�\)C@ ����@z�?�(�A�
=C
=                                    By�  �          @��
��  @33@��A�Q�C����  @   ?�A�Q�C=q                                    By��  �          @����R?�(�@�A���C ޸���R@
=q?�A��HCxR                                    By�R  �          @�����
=?���@  A�\)C#8R��
=@33?�=qA���C�
                                    By�,�  �          @��R��  ?��R@z�A��
C#u���  ?�p�?�A�33Cp�                                    By�;�  �          @��H��
=?�{@�A�33C$����
=?���?�A�
=C�f                                    By�JD  �          @�Q���(�?��H@A�ffC#J=��(�?�33?ٙ�A�  C��                                    By�X�  �          @������?�  @�A�Q�C"�
����?�?��A���C�                                    By�g�  �          @�=q��p�?��H?�\)A�
=Cٚ��p�@?�Q�Az�HC�)                                    By�v6  �          @�z����?�G�?��A��\C"�����?�33?�  A��Cn                                    By���  �          @����?��?�(�A�ffC$J=��?�ff?�{A�Q�C�q                                    By���  �          @�
=��33?��@ffA��C#����33?�=q?޸RA�=qCO\                                    By��(  �          @����G�?���@��A��\C$=q��G�?�ff?��A�{Ch�                                    By���  �          @�
=���?h��@   A��C)����?�=q?޸RA�33C%�                                    By��t  �          @��R���?aG�@G�A��C*  ���?��?�\A�G�C%O\                                    By��  �          @�����z�?���@��AҸRC$+���z�?�@�A��C��                                    By���  �          @�
=��?�z�@$z�A���C����@(�@�A�G�C&f                                    By��f  �          @��
��?�G�@{A�p�CW
��@��@G�A���C�                                    By��  �          @������?�(�@"�\A��
C������@�R@A��C
                                    By��  �          @����33?�\@$z�A�C�)��33@33@�A�ffC0�                                    By�X  �          @��H��{?��@'
=A��
C#����{?�=q@��A�  C�\                                    By�%�  �          @��\���?��@p�AۮC'  ���?�=q@
=qA��C!�                                    By�4�  �          @�z����H?}p�@'�A�
=C'�)���H?��@A�{C!z�                                    By�CJ  �          @�p���(�>�  @7
=A�33C1���(�?^�R@/\)A癚C)�
                                    By�Q�  �          @�\)��(�>u@?\)A��
C133��(�?c�
@7�A�=qC)��                                    By�`�  �          @�\)���>aG�@<(�A���C1z����?Y��@4z�A��C*+�                                    By�o<  �          @����������
@J=qB33C5�����?(�@G
=BG�C,c�                                    By�}�  �          @�\)��33�W
=@U�B��C6Ǯ��33?�@S33B�
C-=q                                    By���  �          @��R�������
@W
=B�RC8=q����>�
=@VffB(�C.z�                                    By��.  �          @��R��Q��ff@B�\B��C9����Q�>aG�@Dz�B�HC1.                                    By���  �          @�  ��녿��@B�\B��C:�)���=�@E�BQ�C2��                                    By��z  �          @����녾���@A�B=qC9����>��@C33B
=C0                                    By��   �          @�G���ff=�Q�@:�HA���C2���ff?5@5A�ffC+c�                                    By���  �          @�Q���(�>�
=@<(�B\)C.���(�?��@1�A�G�C'O\                                    By��l  �          @�Q������G�@2�\A�C5L����>��H@0  A�C.)                                    By��  �          @������\���@C33B=qC9����\>8Q�@E�B��C1��                                    By��  T          @�\)��33�޸R@B�\B	(�CJ���33��ff@UB�CBc�                                    By�^  �          @��������Q�@Q�B33CJ�)�����s33@c�
B"\)CAB�                                    By�  �          @�ff��Q�h��@P��B33C@\��Q�L��@XQ�B�C6�R                                    By�-�  �          @��R���H�\(�@K�B�\C?B����H�8Q�@R�\B\)C6W
                                    By�<P  �          @�  ��
=��\)@J�HB�C7����
=>Ǯ@J=qB�C.�q                                    By�J�  �          @�ff��\)��
=@@��BG�C9T{��\)>aG�@A�BQ�C1+�                                    By�Y�  �          @�=q�����L��@L��Bp�C6�����>��@K�B  C.{                                    By�hB  �          @�����=q�W
=@S33B(�C6����=q>�@QG�BC.                                      By�v�  �          @������\��33@R�\Bp�C8W
���\>�{@R�\Bz�C/��                                    By���  �          @��R��p����H@Q�B�C:
=��p�>L��@S�
Bp�C1��                                    By��4  �          @����H��(�@Tz�B33C9T{���H>�=q@UB
=C0�                                    By���  �          @����\��@<(�A�p�C9n���\>��@>{A�z�C2+�                                    By���  T          @�
=��\)��33@3�
A�C7����\)>u@5�A�\C1@                                     By��&  �          @�����׾���@0��A�(�C7������>u@1G�A�
=C1G�                                    By���  �          @�����(���@6ffA�=qC5k���(�>�@3�
A���C.�
                                    By��r  �          @������\)@Mp�B
{C5����?�@J�HB(�C-�f                                    By��  �          @��������\@N�RB��C:s3����>��@P��Bz�C2)                                    By���  �          @����\)�+�@J�HB\)C<\��\)�#�
@O\)B	C4+�                                    By�	d  �          @�=q��  ��Q�@UB=qC8aH��  >��
@VffBp�C0+�                                    By�
  �          @��H���H���@aG�Bz�C9�\���H>�  @b�\B�\C0�H                                    By�&�  �          @�(���=q=�Q�@Tz�B
33C2�f��=q?B�\@N�RB(�C+
                                    By�5V  �          @����ff���
@N�RB\)C4�3��ff?\)@K�B�C-�                                     By�C�  �          @�(����>�p�@J�HB�C/�����?�  @A�A��C(�=                                    By�R�  �          @�p���
=��(�@b�\B{C9#���
=>�z�@c�
BC0�                                    By�aH  T          @�����
=>8Q�@Tz�B\)C1ٚ��
=?Tz�@Mp�B�C*                                    By�o�  T          @��R��z�>u@Tz�B(�C1���z�?c�
@Mp�B�C)+�                                    By�~�  �          @��\����>��@S�
B
G�C.\)����?�{@I��B�
C&�H                                    By��:  �          @����?8Q�@O\)B��C+�����?��@B�\A��RC$��                                    By���  �          @�ff��{>�Q�@R�\B�C/����{?�  @J=qB Q�C(�\                                    By���  �          @���>�p�@QG�B�C/����?�  @HQ�A�\)C(}q                                    By��,  �          @����
?�\@Tz�Bp�C.����
?��@J=qB ��C&�=                                    By���  �          @����=q?.{@N�RBG�C+�q��=q?��@B�\A���C%�                                    By��x  �          @����  ?��
@<(�A�C(Q���  ?�=q@,(�A�
=C"s3                                    By��  �          @�����=q?�ff@33A��\C#��=q?��H?��HA���C�q                                    By���  �          @�G����@  @33A�
=C0����@)��?���A���C�\                                    By�j  �          @�33����@8Q�@33A�{C=q����@N{?��HAg\)CT{                                    By�  �          @������
@l(�?�{A|  C�=���
@{�?aG�AQ�C
�                                    By��  �          @����@n{?�G�A�33C  ��@\)?��\A\)C

=                                    By�.\  T          @����(�@Z�H?�{A�
=C�{��(�@mp�?�
=A3�CT{                                    By�=  �          @�=q��(�@\��?���A��\CG���(�@o\)?�A2=qC�                                    By�K�  �          @��
��(�@E�?�p�A�\)C� ��(�@Y��?�{AN�HC+�                                    By�ZN  �          @�33��{@8��@ffA���C����{@N�R?\Af�\C��                                    By�h�  �          @�33��
=@,��@�\A�(�C���
=@Dz�?޸RA��HC@                                     By�w�  �          @��
��=q@�@ ��A��CY���=q@0  @�A�G�C��                                    By��@  �          @��
��=q?�@5A�(�C ���=q@@��A���C=q                                    By���  �          @Å���?�
=@<��A��C$\)���?���@(��A��C)                                    By���  �          @������@p�@%�A�ffC�����@8��@A��\C
                                    By��2  �          @Å��(�@�
@%A��HC#���(�@   @
�HA��\C
=                                    By���  T          @�ff���
@ff@p�A�C=q���
@0��?��RA���CxR                                    By��~  T          @�
=���@C33@
=A��
C���@X��?\AlQ�C                                      By��$  �          @�
=����@1�@��A�C�
����@HQ�?�z�A��
CǮ                                    By���  �          @�
=��ff@(�@{A�p�C��ff@333?޸RA�z�C�                                     By��p  �          @�(���=q@[�@�A��
C���=q@o\)?���AY��C
                                    By�
  
�          @��\��{@E�@
�HA��RC8R��{@Z�H?�=qA{�
CaH                                    By��  �          @�(���G�@\)@#33AͮC���G�@*=q@ffA�(�C�R                                    By�'b  �          @�z�����@{@(Q�Aԣ�C
=����@)��@(�A�\)C�                                    By�6  �          @�(���  @G�@2�\A�33C���  @�R@��A�(�Cc�                                    By�D�  �          @�z����
@�@-p�A�p�CG����
@7�@\)A���C.                                    By�ST  T          @�����=q@ff@)��A�Ch���=q@"�\@\)A��\C=q                                    By�a�  �          @�������@�H@p�A��C#�����@3�
@   A�ffC�                                    By�p�  �          @��
��z�@+�@��A�
=C�R��z�@C�
?�33A���C��                                    By�F  �          @��
����@(Q�@�A�p�C�{����@@��?�
=A�  C#�                                    By���  T          @��H��(�@'�@��A���C�\��(�@?\)?�33A�Q�C+�                                    By���  �          @�����33?�G�@Q�A�z�C ���33@	��@�\A�  C.                                    By��8  �          @����  ?�@!�A��C�\��  @\)@
�HA�33C��                                    By���  �          @��
���R@-p�@
�HA�ffC)���R@B�\?�
=A�p�C.                                    By�Ȅ  �          @����z�@1G�@Q�A��C���z�@E?�\)A�  CG�                                    By��*  �          @����@4z�@�A��C����@H��?�\)A�  C!H                                    By���  �          @�G�����@p�@
=qA��C� ����@1�?�(�A�{C�R                                    By��v  �          @�Q����R@N�R?�{A���C{���R@\��?�ffA(z�C:�                                    By�  �          @�p���(�@c�
?�G�Aw
=C:���(�@qG�?c�
A��C	��                                    By��  �          @�z����\@2�\@�A��C=q���\@G
=?�Q�A�Q�CQ�                                    By� h  �          @�(���ff?�z�@$z�A�(�C �\��ff@�
@��A�{CQ�                                    By�/  T          @�\)��33?��@5A��C'Q���33?��@'�AٮC"W
                                    By�=�  �          @������?L��@A�A���C*�����?��@7
=A�RC%(�                                    By�LZ  �          @��\����?@  @P��B�
C+!H����?��\@EB 
=C%#�                                    By�[   �          @���\)?}p�@?\)A��
C(:���\)?��H@2�\A��C"�)                                    By�i�  �          @�
=���?z�H@5A�\C(�
���?�
=@)��Aܣ�C#��                                    By�xL  �          @��H��33?��@C33A�{C'\)��33?���@5�A�z�C"�                                    By���  �          @��\���R?�=q@J�HB�RC$=q���R?���@:=qA��C�)                                    By���  �          @������H?�33@P��B�
C#����H?�33@?\)A��RCs3                                    By��>  �          @�����G�?��
@L��B
=C!W
��G�@G�@:�HA�\)C��                                    By���  �          @�������?��@J=qB=qC#ff����?�\)@9��A�Q�C)                                    By���  �          @�����H?��H@O\)B�HC"T{���H?��H@>{A�ffC�                                    By��0  
�          @�����
=?�p�@EBG�C%p���
=?ٙ�@7
=A�33C @                                     By���  �          @�������?��
@8��A��C(�����?�(�@,(�A�{C#B�                                    By��|  �          @�=q��{?�33@6ffA�\C&�3��{?�=q@(��AׅC"T{                                    By��"  �          @�����?��H@7
=A��HC"�����?��@&ffA���C��                                    By�
�  �          @��
���?�Q�@8Q�A�=qC"�3���?�{@'�A�Q�C                                    By�n  �          @�
=��\)?�Q�@/\)A��C ^���\)@@��A���C0�                                    By�(  �          @�Q���G�?�@.{A�C �\��G�@�
@(�A�Q�C�3                                    By�6�  �          @��R��Q�?�
=@2�\A��C#L���Q�?�@"�\A�  C�                                    By�E`  �          @�ff���\@�@)��A�33C
���\@=q@�
A��HCE                                    By�T  �          @�\)���
@
=q@'
=A؏\C�q���
@!G�@  A�G�C^�                                    By�b�  �          @�33��ff?�Q�@0��A�33C#Ǯ��ff?�=q@ ��A˙�C�f                                    By�qR  �          @�����Q�?��@333A���CxR��Q�@G�@\)A\C��                                    By��  �          @�=q����?�(�@1G�A�p�C�)����@
=@��A�z�C�H                                    By���  �          @�����\@��@,��A��C�����\@$z�@ffA�33C0�                                    By��D  �          @�p�����@{@0  A��
CQ�����@%@��A��HC�{                                    By���  �          @�z����@Q�@0  A���C5����@ ��@=qA��C��                                    By���  �          @�
=��p�?��R@4z�A�{C����p�@Q�@ ��A�{CB�                                    By��6  �          @�
=��ff?�z�@5�A�{C޸��ff@�\@!G�A�G�C5�                                    By���  �          @ƸR��{?�\@:=qAߙ�C!#���{@
�H@(Q�A�(�CL�                                    By��  �          @�ff��{?�(�@:=qA�{C!����{@�@(��A�G�C�\                                    By��(  �          @�p�����?�(�@<(�A�C!������@�@*�HA���C��                                    By��  �          @�ff��(�?�
=@C�
A�\C!�H��(�@@2�\A�(�C��                                    By�t  �          @ə���Q�?��@P��A�G�C'Ǯ��Q�?˅@C�
A��C#=q                                    By�!  �          @��H��  ?�G�@Tz�A��HC&}q��  ?��H@G
=A�p�C!�                                    By�/�  �          @����{?��@UA�\)C%�\��{?�G�@G�A�G�C!@                                     By�>f  �          @�G���ff?�33@H��A��C$B���ff?���@:�HA�C޸                                    By�M  �          @�����H?��H@(��A���C  ���H@33@ffA�p�C�
                                    By�[�  �          @�=q��=q?�\)@*�HA���C����=q@p�@��A�z�CQ�                                    By�jX  �          @�z���{?�\)@*�HA�=qC ���{?�(�@�A��C@                                     By�x�  T          @������?У�@*�HA�\)C �f����?�p�@�HA��HC��                                    By���  T          @�\)��ff?��R@B�\B �C%B���ff?У�@6ffA�=qC �
                                    By��J  �          @�����{?��
@I��B�C$����{?�Q�@=p�A�G�C (�                                    By���  �          @��H����?��\@Dz�A�  C%8R����?�z�@8Q�A�=qC �f                                    By���  �          @������?��@E�A��
C$p�����?�p�@8Q�A�\)C .                                    By��<  �          @�z�����?���@@  A�p�C!������?�p�@1G�A�ffC�{                                    By���  �          @����G�?�33@>�RA�z�C#����G�?��
@1G�A�C�H                                    By�߈  �          @�����\)?�(�@4z�A�\C����\)@z�@%�A�ffCQ�                                    By��.  �          @��\��\)?�Q�@<��A�33C ^���\)@33@-p�A݅C��                                    By���  �          @�33��  ?��@?\)A�=qC �3��  @ ��@0��A�33C{                                    By�z  �          @�33���?���@B�\A��\C!^����?�(�@3�
A�  Ck�                                    By�   �          @��H���R?�=q@Dz�A��C!u����R?���@5A�\)Cu�                                    By�(�  �          @��\��ff?��R@G
=B ��C"xR��ff?�{@9��A�\Cc�                                    By�7l  
�          @�Q���(�?�@HQ�B��C"�f��(�?�ff@;�A�ffC�R                                    By�F  �          @��R����?�(�@I��B��C"�����?���@<(�A�{C�
                                    By�T�  �          @�{��  ?��@N�RB
p�C#����  ?�
=@C33B��C�=                                    By�c^  �          @�������?�p�@H��B\)C$�����?���@>{A��C u�                                    By�r  �          @������R?���@P  B��C$�f���R?˅@E�BG�C c�                                    By���  �          @��H����?�ff@Z=qB=qC&J=����?���@P  Bz�C!Y�                                    By��P  �          @�p���  ?c�
@hQ�B�\C(.��  ?���@`  B��C"Ǯ                                    By���  �          @�ff����?.{@��\B-{C*�����?�z�@~{B'Q�C$��                                    By���  �          @�p����?�@��B333C,�=���?�G�@�=qB.G�C&J=                                    By��B  �          @����~{?L��@s33B,�RC(�\�~{?��R@k�B&  C"�)                                    By���  �          @�  ��{?�Q�@uB�\C%���{?У�@k�Bz�C�)                                    By�؎  �          @ƸR��\)?���@l��B\)C!�H��\)?��R@_\)B	  C33                                    By��4  �          @�p����R?�{@p  B�
C#�����R?��
@dz�Bz�CJ=                                    By���  �          @�����33?��\@w
=B  C$�=��33?ٙ�@l(�B�C��                                    By��  �          @�(�����?���@r�\B�HC ������@ ��@eBG�C�                                    By�&  �          @�33����?Ǯ@l(�B33C�����?�(�@`  B\)CJ=                                    By�!�  �          @������?�
=@h��B  C�����@�@Z�HBp�Cu�                                    By�0r  �          @�  �O\)?���@��B{CL��O\)@ff@�A�C                                    By�?  T          @z=q�7�?�\?��HA�Ch��7�?�(�?�G�A�  C�=                                    By�M�  �          @}p��5�?�\)@�HB=qC!H�5�?У�@  B
z�C
                                    By�\d  �          @g��$z�?.{@�HB){C%.�$z�?p��@�B!��C�3                                    By�k
  �          @p  �$z�?u@"�\B+  C���$z�?�(�@�HB!\)C�
                                    By�y�  �          @{��3�
?#�
@*�HB*��C')�3�
?k�@%�B$ffC!��                                    By��V  �          @w��333?G�@"�\B$z�C$aH�333?�ff@(�B�C}q                                    By���  �          @�  �1G�?�  @#�
B z�C���1G�?�G�@=qB�RC\)                                    By���  
�          @{��7�?xQ�@   BQ�C!B��7�?�p�@��B�RC��                                    By��H  �          @~{�E�?�\@(�B�
C*�)�E�?B�\@�B
=C&(�                                    By���  �          @�G��Mp�?(�@Q�Bz�C)=q�Mp�?Y��@�
B=qC%#�                                    By�є  �          @���HQ�?�@'
=B��C*���HQ�?J=q@"�\B  C%�f                                    By��:  �          @�z��I��?��@(Q�B��C*
�I��?O\)@#�
B�HC%�                                    By���  �          @�z��N�R?Q�@(�B��C%�q�N�R?��@B
33C!��                                    By���  �          @���[�?Y��@/\)B{C&��[�?�\)@(��B��C!�                                    By�,  
�          @����n�R?�{@��A��C���n�R?˅@��A�
=C�                                    By��            @�\)�fff?���@33A�
=CaH�fff?���@
=qA�(�Cff                                    By�)x  "          @�ff�g�?u@��B�RC%��g�?�Q�@33A��RC!�                                     By�8  	�          @��j=q?^�R@z�A�p�C&�R�j=q?�=q@�RA�{C#��                                    By�F�  T          @�p��n�R?.{@��A�{C)���n�R?c�
@�A��C&�)                                    By�Uj  
�          @���o\)?.{@��A�G�C)���o\)?aG�@z�A�ffC&�q                                    By�d  
�          @�ff�p  ?�@�
A�33C,��p  ?=p�@  A�{C(��                                    By�r�  �          @���n{?z�@�\A��RC+#��n{?J=q@�RA���C(�                                    By��\  T          @��\�vff?&ff@�A��
C*aH�vff?^�R@33A�C'J=                                    By��  
�          @��
��G�>���@p�A��C/����G�?�\@�A���C,޸                                    By���  �          @�
=�xQ�?k�@   B G�C&���xQ�?�33@=qA��
C#��                                    By��N  
�          @�=q�r�\?
=@�HB �C+
�r�\?O\)@ffA�ffC'��                                    By���  
�          @�z��l(�>��?��Aי�C1���l(�>��R?�{A��C/(�                                    By�ʚ  �          @xQ��`�׽�?��AǅC5��`��<�?��A�{C3�{                                    By��@  �          @q��Vff    ?�(�A�p�C4�Vff>��?��HA؏\C1s3                                    By���  T          @�=q�p�׾�  ?��
A��C7�{�p�׽�?�ffA�
=C5��                                    By���  
Z          @���hQ�>L��?�Aޣ�C0��hQ�>�Q�?��Aۙ�C.Y�                                    By�2  
�          @��H�dz�>�  ?�Q�A�\)C/�3�dz�>�
=?�z�A�C-Y�                                    By��  
�          @���_\)>�  @�A��C/���_\)>�
=@ ��A�p�C-&f                                    By�"~  "          @��\�`��>�{@�
A��C.ff�`��?�\@�A�  C+��                                    By�1$  
(          @�=q�^�R>���@�
A�C-p��^�R?�@G�A�\C*�R                                    By�?�  �          @�p��[�?\)@33B�
C*���[�?@  @\)B\)C'��                                    By�Np  
�          @�
=�N�R?8Q�@'
=B��C'c��N�R?n{@"�\BffC#�                                    By�]  �          @����[�?
=@   B�C*:��[�?J=q@(�B{C'\                                    By�k�  "          @�Q��U?
=q@%B�C*�{�U?=p�@"�\BG�C'z�                                    By�zb  "          @���aG�>�33@p�BQ�C.c��aG�?
=q@�B	{C+Q�                                    By��  
�          @��
�`  ?�@ffA��HC+p��`  ?0��@33A���C(��                                    By���  �          @���`  ?
=?��HA�\)C*aH�`  ?=p�?�z�A�RC(�                                    By��T  
�          @�z��\��?�@{B�HC*�
�\��?=p�@
�HA�G�C'��                                    By���  �          @���_\)?��@�A�z�C*L��_\)?B�\@Q�A��C'�q                                    By�à  
�          @����\��?��@\)B�C*��\��?5@(�A�33C(J=                                    By��F  	�          @�(��e?&ff?�z�A��HC)��e?J=q?���A�=qC'�f                                    By���  �          @�=q�`  ?z�?�(�A���C*���`  ?8Q�?�
=A�RC(ff                                    By��  T          @�G��^{>�G�@   A�=qC,�3�^{?
=?��HA�\)C*ff                                    By��8  �          @��H�e?��?��A�
=C+B��e?.{?�AՅC)8R                                    By��  "          @���a�?
=?�33A�p�C*z��a�?8Q�?���Aٙ�C(k�                                    By��  
�          @�=q�[�?#�
@A���C)c��[�?J=q@�\A�Q�C'�                                    By�**  
(          @����Vff?(�@Q�B G�C)�)�Vff?B�\@�A�=qC':�                                    By�8�  
�          @���\��?G�?�p�A��
C'Q��\��?h��?�A�z�C%@                                     By�Gv  �          @~�R�Vff?&ff@33A��C(���Vff?J=q@   A�
=C&�q                                    By�V  "          @y���O\)>�
=@�B�
C,���O\)?\)@B�\C*:�                                    By�d�  "          @q��K�>�(�@   A�33C,L��K�?\)?��HA��\C*                                    By�sh  �          @s�
�I��>�  @�B\)C/c��I��>Ǯ@ffB�
C,�                                    By��  �          @u��Mp�����@�Bz�C5�)�Mp�<�@�B��C3�                                     By���  |          @|(��U��@G�A��C<33�U��33@33A��RC:�                                    By��Z  
�          @w
=�Fff�L��@�\B��C4�)�Fff=���@�\B�C2:�                                    By��   T          @xQ��G
==L��@z�B��C3#��G
=>B�\@�
BG�C0��                                    By���  "          @k��7
=>��@33B�HC.���7
=>���@�BQ�C,\                                    By��L  
�          @s33�A�?@  @
=B�C&��A�?^�R@�
B  C#�f                                    By���  
�          @tz��=p�=�G�@=qB�\C1���=p�>�  @��BC/.                                    By��  �          @~{�*=q?�@#�
B!p�C��*=q?���@�RB��C��                                    By��>  
�          @`  �   >W
=@�B0��C/(��   >�33@�HB/�C,                                    By��  
Z          @g��!G���@   B0�HC>�!G���{@!G�B2�C;��                                    By��  
(          @o\)���?=p�@2�\BH�
C!}q���?c�
@0  BDffC��                                    By�#0  T          @A녿�\)?���@B*�HC���\)?�Q�@   B!��Cٚ                                    By�1�  T          @ �׿�ff>k�?�(�B.
=C,����ff>��
?��HB,(�C)��                                    By�@|  T          @-p���(�>�G�?�(�B?�C%�
��(�?��?�Q�B<(�C"c�                                    By�O"  
(          @%���z�?p��?�BBp�CB���z�?��?�\)B;�C��                                    By�]�  �          @ �׿��?�  ?�{B@ffCG����?��?�ffB9�C                                    By�ln  �          @ �׿�?p��?�B<�RC�{��?��
?��B6{C\                                    By�{  T          @����?.{?��BB�RCT{��?E�?�G�B=z�C}q                                    By���  
�          @(����R?��?���BB{C\���R?0��?��B=��C.                                    By��`  "          @-p�����?@  @�\BDC�����?Y��@   B?C                                      By��  �          @������?=p�?��BC�C������?Tz�?�  B=�RC
=                                    By���  �          @(���Q�?Q�?��
B;�HC\)��Q�?fff?޸RB6\)C�                                    By��R  
�          @�Ϳ���?!G�?˅B9Q�C�׿���?333?ǮB4C0�                                    By���  "          @Q쿢�\?:�H?��
B9=qC.���\?L��?�  B3�HC�f                                    By��  �          @ff����?
=?��HB>��C�����?+�?�Q�B:�\C=q                                    By��D  �          @��G�?z�?�z�B6C&f��G�?&ff?У�B2��C��                                    By���  T          @�\��{>�ff?�Q�B4��C!��{?�\?�B1\)Cs3                                    By��  �          @�ÿ���?
=q?��B:�C0�����?��?\B7(�C�)                                    By�6  �          @���z�?
=q?޸RBDG�C\��z�?(�?�(�B@�C�\                                    By�*�  �          @���H?   ?޸RBA�C!����H?�?�(�B>�RC�
                                    By�9�  
(          @���У�?
=q?��B/�C!�ͿУ�?��?�\)B,�C�                                     By�H(  "          @z῱�>��?��HB4ffC#� ���>��?�Q�B1�C!aH                                    By�V�  "          ?��R���\?#�
?�=qB,
=CB����\?0��?�ffB'��CxR                                    By�et  "          @�
��{?8Q�?�{B&33C���{?E�?�=qB"
=CxR                                    By�t  "          ?��R���?z�?���B*Q�C�����?!G�?���B&�HC�)                                    By���  �          ?�
=����>��H?��
B)(�C�ÿ���?�?�G�B&(�C��                                    By��f  
Z          ?�33��33>�p�?�{B)��C"E��33>��?��B'  C �=                                    By��  �          ?���=q>���?���B.�HC%�ÿ�=q>\?�ffB,�
C$&f                                    By���  T          @�\��(�>k�?�{B)�HC+Ϳ�(�>�\)?���B(�\C)Q�                                    By��X  "          ?�Q쿇�>�(�?�p�B;�C����>��?��HB8��C+�                                    By���  
(          ?���=q>�p�?�Q�B833C �3��=q>��?�
=B5�C+�                                    By�ڤ  �          ?\�s33>\?�{B=�Cuÿs33>��?��B:\)C��                                    By��J  �          ?���33>�{?��B.  C#\)��33>\?���B+�HC!Ǯ                                    By���  �          ?�
=��G�>�z�?��B�\C&�{��G�>���?��
B�C%�                                     By��  �          ?�Q쿠  >�  ?���B"��C(����  >�\)?��B!=qC'E                                    By�<  �          ?��H��>�?��RB��C!c׿�?�\?�p�B��C !H                                    By�#�  �          ?޸R���>�  ?��B��C)�{���>�\)?�=qB��C(W
                                    By�2�  �          @�
���?L��?��B \)CG����?Tz�?��B(�C0�                                    By�A.  �          @   ���?E�?�G�BQ�CǮ���?L��?��RB=qC��                                    By�O�  �          @Q쿸Q�?L��?��B��C޸��Q�?W
=?���BCٚ                                    By�^z  �          @녿�  ?xQ�?�33BffC���  ?�G�?�\)BG�C�                                    By�m   �          @+����?���?У�B  Ck����?��?���B�C��                                    By�{�  �          @B�\���?�G�?���B��CO\���?�ff?�BG�C��                                    By��l  �          @@  ���H?�ff?�BQ�CLͿ��H?��?�Bp�Cz�                                    By��  �          @5���G�?���?�  B�HC���G�?��?�(�B�
C                                    By���  �          @1녿�G�?�=q?�Q�B\)C�)��G�?�\)?�z�Bp�C#�                                    By��^  �          @.�R��{?�
=?�z�BC
n��{?��H?У�B�C	Ǯ                                    By��  
�          @H�ÿ�  ?�
=?���BG�C5ÿ�  ?��H?�B(�C��                                    By�Ӫ  �          @R�\��p�?�{@�
B�C�)��p�?�33@�B��C33                                    By��P  �          @[���R?�{@�
B33C.��R?�33@�BC��                                    By���  �          @?\)��p�?�  ?��B��C���p�?��
?�\B��C�                                    By���  �          @*=q��ff?��?�=qB��C����ff?�?���B��C�3                                    By�B  �          @^{�"�\?�\)?�33Bp�C�{�"�\?�33?��B�RC�                                    By��  �          @r�\�?\)?�z�?�{A���C�3�?\)?�Q�?�A�  CB�                                    By�+�  �          @e���
?˅@	��B�RCff��
?�\)@Q�B�
C��                                    By�:4  �          @S�
��\?��?���B\)C����\?�z�?�Q�B�RCW
                                    By�H�  �          @   ��?k�?�G�A�G�C+���?n{?�  A��HC޸                                    By�W�  �          @{��
?c�
?}p�A�z�C����
?fff?z�HA�=qCs3                                    By�f&  �          @��z�?s33?p��A��\Cff��z�?xQ�?n{A�(�C#�                                    By�t�  �          @p�� ��?��?s33A�33C� ��?�ff?n{A���C�=                                    By��r  �          @!��?�{?\(�A���C��?�\)?Y��A�z�C                                    By��  �          @*�H��(�?�ff?�Q�A�ffC�)��(�?��?�
=A�  CaH                                    By���  �          @0  ��?�ff?�p�A�z�C{��?��?��HA�ffC޸                                    By��d  �          @7
=��z�?��?�Q�B
=C{��z�?�ff?�
=B
=C�\                                    By��
  �          @.{��=q?�{?�
=B�HC�{��=q?�\)?�B  C��                                    By�̰  �          @-p���?��?�Q�B�Cz��?���?�
=B\)C:�                                    By��V  
�          @@  �z�?�ff?ٙ�B�C�{�z�?��?ٙ�B
��C�H                                    By���  �          @;��G�?��
?��HA�C���G�?��?���A�Q�C�                                    By���  �          @1G���{?�z�?�p�B�RC�q��{?�?�p�B�C�)                                    By�H  �          @1녿�=q?�Q�?�G�B  CǮ��=q?���?�G�Bz�C��                                    By��  �          @0  ��{?���?�p�Bz�Cp���{?���?�p�B{C\)                                    By�$�  �          @(�ÿ���?���?�Q�B=qC�ÿ���?��H?�
=B��C��                                    By�3:  �          @ �׿��?�G�?�
=B
�C�����?�G�?�B
�RC��                                    By�A�  �          @   ��ff?Tz�?�G�B�
Ch���ff?Tz�?�  BC^�                                    By�P�  �          @
=���?�?�
=BC#�H���?�?�
=BC#޸                                    By�_,  �          @
=��(�?0��?���B(�C���(�?0��?���B33C�                                    By�m�  �          @�R�Ǯ?W
=?���BG�C� �Ǯ?W
=?���BffC�=                                    By�|x  T          @G�����?h��?���B(�C:����?h��?�=qB\)CL�                                    By��  �          @���Q�?n{?�ffB�\C33��Q�?n{?�ffB��CJ=                                    By���  �          @���p�?aG�?���B
�
C�3��p�?aG�?���B(�C\                                    By��j  �          @{��  ?u?�Q�Bz�C���  ?u?���B�CB�                                    By��  �          @���?u?�(�B�HC(���?s33?�p�B\)CT{                                    By�Ŷ  �          @!G���ff?�ff?�z�B{C�ÿ�ff?��?�z�B�C�H                                    By��\  �          @{��G�?�G�?�BffC
��G�?�G�?�B
=CJ=                                    By��  �          @������?�=q?���A�z�C�{����?���?�=qA��C                                      By��  �          @�ÿ�G�?�\)?�{A��Ch���G�?�{?�\)A��HC��                                    By� N  �          @��У�?��?�33B��C��У�?���?�z�B  CE                                    By��  �          @���ff?��?�Q�B��C���ff?��
?���B�RCk�                                    By��  �          @   ����?���?ٙ�B+33C  ����?�{?��HB,�Cc�                                    By�,@  �          @{��33?�  ?���B"G�C
@ ��33?�p�?�\)B#C
�)                                    By�:�  �          @/\)��33?J=q?��B#��Ck���33?E�?��B$�\C��                                    By�I�  �          @%���?z�H?�  B�HC��?u?�G�B{CW
                                    By�X2  �          @G���p�?Y��?�
=A�{C�H��p�?W
=?�Q�A�z�C@                                     By�f�  �          @�Ϳ���?G�?s33A���C�q����?E�?uA�
=C\                                    By�u~  �          @p�����?=p�?�  AָRC�����?:�H?�G�A���C�                                    By��$  �          @�Ϳ�?E�?h��Aď\C=q��?B�\?k�A���C��                                    By���  �          @�׿�=q?\(�?�G�A�ffC޸��=q?W
=?��\A�33C=q                                    By��p  �          @=q��?�G�?��A�p�C(���?�  ?��AЏ\C�                                    By��  �          @Q����?�  ?��A�\C8R����?z�H?�z�A�  C�f                                    By���  �          @
=��(�?s33?Y��A�
=C8R��(�?p��?\(�A�=qC�=                                    By��b  �          @{����?n{?8Q�A���CaH����?k�?:�HA�=qC�                                    By��  �          @Q���?+�?0��A�  C uÿ��?(��?333A���C �=                                    By��  �          @\)�z�H?�?��HBP{C�)�z�H?��?�p�BS33C��                                    By��T  �          @�Ϳ��?�{?�z�BM�RC33���?���?�
=BP�
C#�                                    By��  �          @   ��
=?���?�33BFQ�C	Ϳ�
=?���?�BIQ�C
                                    By��  �          @ �׿�(�?��?��BDG�C
8R��(�?�ff?�z�BGG�C8R                                    By�%F  �          @{����?�\)?���BA��C������?�=q?�\)BD��C	�f                                    By�3�  �          @\)���?���?���B<�C�
���?��
?�B?(�C޸                                    By�B�  �          @ �׿���?s33?�p�BQ(�CǮ����?h��@   BT33C�                                    By�Q8  �          @{��{?fff?�p�B(�HCh���{?^�R?�  B+�Cc�                                    By�_�  �          ?�(���z�?B�\?:�HA�{Cz��z�?=p�?=p�A�ffC�R                                    By�n�  �          ?��R��(�?O\)?�A��RC�H��(�?J=q?z�A�G�C@                                     By�}*  �          @���?G�?333A�  CǮ��?B�\?8Q�A�Q�C=q                                    By���  �          @G���?n{?s33A�=qCJ=��?fff?xQ�A�G�C�H                                    By��v  �          @�Ϳ�z�?��?��AӮC���z�?��
?�{A�33C�f                                    By��  �          @�H����?�{?���A��HCͿ���?�=q?�\)A���C�3                                    By���  �          @����?�\)?c�
A�  C�=��?���?k�A�  CQ�                                    By��h  �          @�׿�{?��?�AZ�HC�׿�{?�\)?\)Ag�C�R                                    By��  �          @
=��33?�
=?.{A�{C���33?�z�?8Q�A��RC�                                    By��  �          @�R��(�?Q�?�33A�z�Ck���(�?J=q?�A�  CO\                                    By��Z  �          @p���?�?�{A�p�Cff��?��?��A�z�C�                                    By�   �          @(���?���?fffA�ffCc׿�?��?p��A˅C\                                    By��  �          @33��p�?���?z�HA˙�C���p�?���?��\A�
=C�                                    By�L  �          @�׿�\)?�(�?xQ�A���C#׿�\)?�Q�?�G�A��C�{                                    By�,�  �          @{��(�?��?��AݮC.��(�?�ff?�Q�A�Q�C�                                    By�;�  �          @p���33?�{?���A�C� ��33?���?��RA��CL�                                    By�J>  �          @����=q?�\)?�z�A�C  ��=q?�=q?���A�\)C�=                                    By�X�  �          @�\��Q�?�=q?��A�33C	O\��Q�?��?�
=A�G�C
#�                                    By�g�  T          @G����
?�z�?��\A�  C	�����
?�\)?��A�(�C
�                                    By�v0  �          @zῥ�?�
=?��B 33C	n���?�33?���B\)C
\)                                    By���  �          @
=��=q?�\)?�
=B�
C޸��=q?�=q?�(�B�
C�                                    By��|  �          @!G���Q�?�{?�33A�z�C+���Q�?���?�Q�A�Q�C
                                    By��"  �          @!녿�\?�(�?��Ař�C&f��\?�Q�?�{A��C��                                    By���  �          @+��   ?�ff?c�
A���C&f�   ?\?p��A��RC�q                                    By��n  �          @   �˅?�{?���A�\)CǮ�˅?���?�z�A��C��                                    By��  �          @$z��\)?У�?�=qA�\)C���\)?˅?��A�\)C�                                     By�ܺ  �          @L���
�H@�?�ffA�  C#��
�H@�\?���A��
C��                                    By��`  �          @\���
�H@ ��?޸RA�Q�C	
�
�H?���?���B 
=C
�                                    By��  �          @Vff���?�(�?��A�{C
@ ���?�33?�{A�C#�                                    By��  �          @Q����?�Q�?��A�{CL����?��?�AͮC
                                    By�R  �          @P  ���?��H?\(�Aw�C�H���?�
=?n{A�G�C#�                                    By�%�  �          @>{��?�=q>L��@w�C���?���>��@��
C�                                    By�4�  �          @J=q�
=@33?�\A\)C)�
=@G�?
=A+�Ck�                                    By�CD  �          @XQ��=q@>�z�@�\)C�f�=q@z�>Ǯ@��
C{                                    By�Q�  �          @B�\���@ff?�\A=qC}q���@�?��A4��C�\                                    By�`�  �          @7
=��z�@�R?�ffA�G�B�\��z�@�?��A��B���                                    By�o6  �          @:=q�ٙ�@Q�?�  A��
C ��ٙ�@�?��A�
=C+�                                    By�}�  �          @-p���\)@ ��?G�A��HC �H��\)?�(�?^�RA�(�Cc�                                    By���  �          @��\)?�{?h��A�Q�CLͿ�\)?���?z�HA�p�C
=                                    By��(  �          @�����?��
?�(�A�=qCY����?�(�?��
B�\Ch�                                    By���  �          @3�
�Ǯ?�?���A�G�C!H�Ǯ?�{?�z�A�
=C
=                                    By��t  �          @,(���=q@?��A��B��q��=q@�\?�33A�
=B�                                      By��  T          @"�\�Tz�@��?O\)A�ffB܀ �Tz�@{?h��A�  B�(�                                    By���  �          @)����{@G�?��\A�G�B��΅{?�(�?�{AƸRB�L�                                    By��f  �          @(�ÿ���@��?c�
A��B������@?}p�A�{B�                                    By��  �          @#33�}p�@  ?��A\��B�aH�}p�@{?5A�=qB���                                    By��  �          @/\)��ff@�?��HA�(�B�  ��ff@G�?��A�\B��                                    By�X  �          @$zῐ��@�
?��
A�=qB�{����@ ��?�\)A�p�B���                                    By��  �          @   �J=q@\)?B�\A�G�B��
�J=q@p�?^�RA�(�B�u�                                    By�-�  �          @#�
���@��?\(�A���B����@ff?uA��RB�                                      By�<J  �          @1G���G�@{?��\A���B�{��G�@
�H?���A\B�G�                                    By�J�  �          @'
=���\@
�H?�G�A���B�G����\@�?�\)A�G�B�aH                                    By�Y�  �          @%���ff@�
>��AffB�
=��ff@�\?�A<Q�B�z�                                    By�h<  �          @"�\�fff@��G���HB��)�fff@ff<#�
>�z�B���                                    By�v�  �          @ff���H@ ��>8Q�@�ffBπ ���H@   >�\)@�G�BϨ�                                    By���  �          @����@�
=#�
?z�HB��Ϳ���@33>��@g
=B��f                                    By��.  �          @z�s33@
=>.{@��\B䙚�s33@ff>�\)@�G�B��
                                    By���  �          @\)�+�@ff���Mp�B�p��+�@ff�#�
��{B�aH                                    By��z  �          @�׿Y��@��W
=���B��Y��@������ ��B��                                    By��   �          @p���?�33>�ffA*�\B�aH��?��?��AQ��B�(�                                    By���  �          @{��  ?�{?xQ�A�{C)��  ?��?��A�
=C8R                                    By��l  �          @!녿�(�?�(�?���AΣ�C�H��(�?�33?�Aޏ\C�
                                    By��  �          @!녿�G�?���?�z�A���C�)��G�?���?�p�A�(�C0�                                    By���  �          @녿�\)?�G�?���B �C���\)?s33?�  B��C�                                    By�	^  �          @  ����?��\?�p�Bz�C  ����?s33?��B=qC�{                                    By�  �          @�\����?�  ?�ffA܏\C�����?�Q�?�{A�(�C�3                                    By�&�  �          @ff��ff?�
=?�  B�C�)��ff?�{?��B
33CT{                                    By�5P  �          @
=��33?s33?�  BC�Ϳ�33?aG�?�ffB�
C޸                                    By�C�  
�          @\)�Ǯ?��?�\)A�C�q�Ǯ?�=q?�Q�B \)CY�                                    By�R�  �          @p���Q�?��?��
B
=C)��Q�?xQ�?��B�\C�                                    By�aB  �          @����
?
=?��BU
=CQ쿣�
>�?�BZ�C��                                    By�o�  �          @Q��=q?k�?�B
=C���=q?W
=?�(�B�\C�                                    By�~�  �          @����R>�ff?�33B\��C�῞�R>�{?�
=Ba
=C$��                                    By��4  �          @�\����?�R?�G�BI=qCǮ����?�?��BN�\C�)                                    By���  �          @!G��\?n{?�G�B2z�C���\?Q�?�B9G�C�
                                    By���  �          @>{��p�?�(�?�B�\CB���p�?�{?��B�C��                                    By��&  �          @g
=�\)>�p�@0��BJ  C*���\)>W
=@1�BK��C.��                                    By���  �          @|���@��?��@�A�(�C�=�@��?�@
=BQ�C��                                    By��r  �          @����O\)@��?s33A\Q�C�=�O\)@�?�=qAz�RCO\                                    By��  �          @c�
�"�\@��    ���
C{�"�\@(�>\)@ffC!H                                    By��  �          @p  �1G�@�����HC�\�1G�@=�G�?��
C�
                                    By�d  T          @w
=�\)@>�R�8Q��0  B��
�\)@AG��
=q��\B�#�                                    By�
  �          @z�H�p�@?\)���H��C� �p�@@�׾�����CE                                    By��  �          @n{�
=@5��.{�'�C�\�
=@5�<#�
>\)C                                    By�.V  �          @l�Ϳ�\@<�Ϳs33�x(�B����\@@�׿E��F�\B��f                                    By�<�  �          @?\)�ٙ�?ٙ�?ٙ�B�HC��ٙ�?˅?�ffB��C��                                    By�K�  �          @>�R�У�?�33@�\B.
=C:�У�?��\@�B7G�C                                    By�ZH  �          @5�����?�\)@��BMCxR����?xQ�@G�BV�RCE                                    By�h�  �          @B�\����@�\?��RA�  B�.����?�Q�?�{B�\C �q                                    By�w�  �          @@  ��Q�@#33�5�b{B�8R��Q�@%�
=q�,Q�B�=                                    By��:  �          @;��5@"�\��  �˙�B�8R�5@'�������HB�W
                                    By���  �          @8�ÿ��\@*�H�������HB�Ǯ���\@,(��#�
�EBݏ\                                    By���  �          @1녿�\)@       =���B�Q쿏\)@\)>.{@b�\B�k�                                    By��,  T          @)����Q�@zᾸQ���(�B�zῘQ�@�W
=����B��                                    By���  �          @����?�ff?n{A���Cn����?�p�?��
AۮC�f                                    By��x  �          ?�z��?�33�W
=��HB�uþ�?����B�\��Bߞ�                                    By��  �          ?���>�=L�Ϳ��H��@�33>�>#�
���H(�A��                                    By���  �          @ ��>W
=?0�׿�.B�k�>W
=?O\)���ǮB�aH                                    By��j  �          @��>W
=?�(������:�B��H>W
=?��ÿ��H�+33B��                                    By�
  �          @��8Q�@
�H�Tz���33B��=�8Q�@{�.{��=qB�Q�                                    By��  �          @"�\����?��ÿ��H�*p�B̀ ����?�
=�˅�(�B��                                    By�'\  �          @"�\���?Q��33=qB�B����?z�H�\)�{p�B��                                    By�6  �          @=q�O\)?
=���C�=�O\)?=p�����{\)C	�                                    By�D�  �          @ff�5?�R�Q�
=C
��5?E����}
=C�                                     By�SN  �          @
=���
?�ff��\�T
=B��)���
?���
=�D��B͔{                                    By�a�  T          @%��(�?W
=�ffB�B���(�?�  ��\�{=qB�u�                                    By�p�  �          @ff��G�?z�H�33�y(�B�  ��G�?��׿�p��jffBޔ{                                    By�@  �          @�\��Q�?�  ��=q�Z  B�{��Q�?��׿�p��J�B�L�                                    By���  �          ?��R�L��?�����  B��׾L��?��;Ǯ�?�
B�k�                                    By���  �          @G�>�z�@�R>L��@��B�  >�z�@p�>�33A  B��H                                    By��2  �          @
�H>u@Q�>��@�33B�.>u@
=>��A.{B�
=                                    By���  �          @�>�  @\)?�AR{B�33>�  @(�?.{A�p�B���                                    By��~  �          @��>�Q�@�\?�AJ{B�(�>�Q�@  ?.{A�p�B���                                    By��$  �          @��>��R@z�?5A�ffB��H>��R@��?^�RA���B��                                     By���  �          @"�\>�{@z�?xQ�A�  B�z�>�{@\)?���Aԏ\B��                                    By��p  �          @p�>��@p�?��A�Q�B�\)>��@Q�?���A��B��
                                    By�  �          @G�>���@�?B�\A���B��H>���@�
?h��A�B�k�                                    By��  �          @>�Q�?���?�
=Bp�B�k�>�Q�?��H?�ffB&B�                                    By� b  �          @	��>�z�?��R?�z�B+
=B�{>�z�?��?�G�B;ffB�u�                                    By�/  �          ?�\)>8Q�?��H?���B  B�.>8Q�?���?�B ��B�aH                                    By�=�  �          @?+�?�\?��
B�B��\?+�?�
=?�z�B��B�W
                                    By�LT  �          @?�?�{?��\Bp�B�� ?�?�G�?�33B��B��
                                    By�Z�  �          @�
?��?���?�(�B 
=B�W
?��?�G�?���B�B��=                                    By�i�  �          @{?.{?�p�?���B�
B�=q?.{?��?�=qB�\B�                                    By�xF  �          ?�{?�\?��?h��A��B�{?�\?�(�?��\BffB�ff                                    By���  �          ?�ff>�(�?��?��B�B��q>�(�?�p�?�Q�B)�
B�ff                                    By���  �          ?�G�>�?z�H?��B/
=B���>�?fff?�{B>ffB{�                                    By��8  �          ?�>��H?k�?p��B'�\Bw�H>��H?Y��?�G�B6��Bo�H                                    By���  �          ?�=q>�?��?�ffB+
=B��H>�?u?���B:�RB�W
                                    By���  �          ?�  >��R?Y��?W
=B+�B�#�>��R?G�?fffB;�
B��                                    By��*  �          ?˅>���?��
?�z�B<�B���>���?p��?�p�BL�B�k�                                    By���  
�          ?�33>�(�?�p�?��B6B�#�>�(�?���?�
=BG  B��
                                    By��v  �          @�;W
=@   ?5A���B���W
=?�Q�?\(�A�\)B�u�                                    By��  �          @�ÿ   @�\>��@��Bϔ{�   @ ��>��A4  B��                                    By�
�  �          @\)���@zᾀ  ��Q�B�{���@��Q���
B��f                                    By�h  �          @�׾��@
=q�W
=���\Bɔ{���@
�H�#�
���
Bɀ                                     By�(  �          ?������?��;�Q�����B����?��׾�\)�F�\B�                                    By�6�  �          ?�z῜(�?�  ��=q�	33C���(�?�=q��  ����C
�                                    By�EZ  �          ?�����?h�ÿ���
=qC������?z�H�u����C��                                    By�T   �          @���ff?u��{� (�C� ��ff?�����
�G�C��                                    By�b�  �          @ff�Ǯ?Y�����\��RCJ=�Ǯ?n{�s33���C.                                    By�qL  �          @��У�?�
=��  �ᙚC#׿У�?�Q�#�
��{C�                                    By��  �          ?�33���\?���?L��A͙�C�Ϳ��\?�G�?aG�A�(�C�\                                    By���  T          ?�33��Q�?���>�Ak\)C����Q�?��?\)A�
=C�                                    By��>  
�          @���ff?��\?�(�B4�C���ff?h��?�ffB@��C(�                                    By���  �          @�
��{?s33?�Q�B2�Cz῎{?Tz�?�  B=Q�C{                                    By���  �          ?�(����R?\)?��
B  C@ ���R>�?���B�C"�                                    By��0  �          ?��Ϳ��\?&ff?�G�B33C�
���\?�?�ffB  C�\                                    By���  �          ?�ff��ff?G�?�RA��\C�ÿ�ff?:�H?.{AĸRC�f                                    By��|  �          ?�  ���#�
�\�J�RC4LͿ�<��
�\�JffC3J=                                    By��"  �          ?��
��G�=#�
�u���C2�)��G�=#�
�L�Ϳٙ�C2�R                                    By��  �          ?��ͿУ׿L�;�\)�33CNE�У׿G���{�*�HCM�{                                    By�n  �          ?�=q�޸R���#�
��ffCCO\�޸R���L����(�CB޸                                    By�!  �          ?�33��=q<#�
�����+33C3�q��=q<������*=qC2��                                    By�/�  �          ?�{��?W
=��(����Cp���?^�R��Q��`Q�Cn                                    By�>`  �          ?�zῇ�?O\)�L�����C�{���?^�R�=p���\)C�\                                    By�M  
�          ?�(��k�?J=q�G��=qCk��k�?W
=�5��Q�C	Y�                                    By�[�  �          ?���5?u����ՅB�8R�5?�  ����{B��q                                    By�jR  �          ?��Ϳ5?����0����\)B�8R�5?�  �
=��B�#�                                    By�x�  �          ?�\)���?=p��333��C𤿋�?J=q�#�
�̸RC
=                                    By���  �          ?ٙ����R?+������Z{C�����R?333��{�;�C�R                                    By��D  �          ?��H��{?\(��\�S
=C��{?c�
���R�+�C�                                    By���  �          ?�(���{?c�
�������C�=��{?k��Ǯ�Xz�CǮ                                    By���  �          ?�p����?k����R�0��C�q���?p�׾u�ffCY�                                    By��6  �          ?ٙ���p�?fff�(�����C�q��p�?s33�
=���C�                                     By���  �          @녿�  ?��H��{����Czῠ  ?���}p��ԣ�C �f                                    By�߂  �          @녿�{?�녿����\CW
��{?�p����\���
C�)                                    By��(  T          @#33��  ?��ÿ\�{C
����  ?�Q쿴z���\C{                                    By���  
�          ?��
�
=q�L�Ϳ0�����Ck�R�
=q�=p��@  �  Ci��                                    By�t  �          ?�z�B�\�z�H���H�1=qCh.�B�\�aG�����?�Ce#�                                    By�  �          ?�33�z῾�R�E���Q�Cx��z῵�c�
���\Cw�                                    By�(�  �          @
=q�E���(���G����\Cu�R�E���녿���{Ctٚ                                    By�7f  �          @p���G���G���Q��4�C`�
��G��c�
��G��A(�C]B�                                    By�F  �          ?�{�k�<��
��{�H�RC2�q�k�=�G������G�C-z�                                    By�T�  �          @Y���У��
=q��z���\)Ch��У��G���=q�G�Cg)                                    By�cX  �          @aG����H��������CgͿ��H��\��G���=qCep�                                    By�q�  �          @��
��\)�J=q��=q��ffCoh���\)�AG���=q��(�CnG�                                    By���  �          @��R��\)�XQ���
�˅CtxR��\)�N�R�33����Csh�                                    By��J  �          @R�\�������p��Dz�CZW
����\)�#�
�OffCU�=                                    By���  �          @e��У׿:�H�<(��j�CL#׿У׾��H�?\)�q33CD�                                     By���  
�          @0�׿��H��G���
�y\)C9#׿��H=��
��
�y��C0@                                     By��<  T          @Tz���
��{����,�HC^&f���
��
=�Q��8�CZ�f                                    By���  �          @����(��(���p���\Ca
=�(��G��
�H�33C_                                      By�؈  �          @fff��=q�@  �p���v�HCrB���=q�:�H��
=��=qCq��                                    By��.  �          @����?O\)��p��?33C�����?k���z��3�C#�                                    By���  �          ?���    ?z�H����33B��H    ?xQ�=L��@?\)B��H                                    By�z  �          ?:�H�W
=?\)�u���
Bܽq�W
=?z�B�\���HB۞�                                    By�   �          ?L�;�(��z����RCi�H��(��녾.{�V�RCiO\                                    By�!�  �          ?�(�������\�aG��z�Ct�������  ���
�O�Ct8R                                    By�0l  �          @	�����ÿ��
��ff�Dz�Co)���ÿ޸R�
=��  Cn�                                     By�?  �          @Q쿏\)��\���Q�Cq!H��\)�G�������33Cp�                                    By�M�  �          @��\(��\)>�  @�{CyͿ\(��  =�\)?���Cy(�                                    By�\^  �          @(��   ����  ��
=C�C׿   �33��
=�3�C�.                                    By�k  T          @&ff    �!녾�Q���RC���    �\)���M�C���                                    By�y�  �          @33���
���:�H���Cp�)���
��\�^�R��(�Co�R                                    By��P  �          @1�>aG��=q>�p�A	�C��{>aG��(�>.{@\)C��                                    By���  �          @Q�?�{��?O\)A��C��=?�{��p�?.{A�z�C��                                    By���  �          @/\)?�{��?5Aqp�C��\?�{��H?�\A.{C�U�                                    By��B  �          @$z�?(��˅?���B;�C��=?(���p�?�
=B*�C��q                                    By���  �          @%>����=q���R��ffC��=>������   �<��C��R                                    By�ю  �          @ff?0���   ?!G�A�33C�xR?0����\>��AD(�C�G�                                    By��4  �          @'�?.{��z�?�=qB  C��\?.{��?�
=Bz�C�C�                                    By���  �          @+�?J=q�33?�ffA��HC�j=?J=q�Q�?\(�A�C�!H                                    By���  �          @E�?#�
�333?Tz�A��RC�n?#�
�7
=?��A9�C�N                                    By�&  �          @W�?!G��'
=@�B\)C��?!G��1G�?�{B\)C�c�                                    By��  �          @X��?(��0  ?�{B�C�Ff?(��8��?У�A�33C���                                    By�)r  T          @XQ�?!G��0  ?���B{C�s3?!G��8��?˅A�{C�'�                                    By�8  �          @K�?G��#�
?��HB��C�l�?G��,(�?��RA�z�C��                                    By�F�  �          @Mp�?���,(�?У�A�  C�Ff?���4z�?�33A��C��                                    By�Ud  �          @A�?0��� ��?�G�A�RC��H?0���'�?�ffA��C�P�                                    By�d
  �          @\)>���?�  A�  C�y�>��
�H?�=qA�{C�4{                                    By�r�  �          ?�(�<#�
����?333A���C�1�<#�
��33?��A��RC�0�                                    By��V  J          @!�?�(���z�?��B%��C�p�?�(����?��
Bp�C�9�                                    By���  
�          @{?�{����?�
=A�{C�n?�{��Q�?��
A�ffC�ٚ                                    By���  ,          @��?�{��G�?�Q�A�{C�)?�{����?�ffA���C�y�                                    By��H  	�          @*�H?��ÿ���?�B(�C��?��ÿ�
=?��\A�Q�C�'�                                    By���            @2�\?��Ϳ˅?���B&=qC�AH?��Ϳ�p�?�(�B{C��                                    By�ʔ  �          @0  ?�\)�p��@�BR��C��?�\)��{@ffBG��C�|)                                    By��:  
�          @"�\?n{���\?�BJ�\C�+�?n{��?�B;�\C��                                    By���  �          @�\?(�ÿ�z�?s33A���C���?(�ÿ�(�?J=qA�ffC�K�                                    By���  �          @(Q�?�{��33?�z�B�RC��?�{� ��?�G�A�C�s3                                    By�,  �          @+�?�z��z�?�  B#��C�w
?�z���?�{Bz�C�o\                                    By��  
�          @;�?��
��=q@BY\)C��f?��
��G�@�RBJ��C�'�                                    By�"x  �          @>{?+���\)@)��B(�C���?+�����@#33BoG�C�w
                                    By�1  
�          @(�?0�׿��H?�(�B)�
C��)?0�׿���?���B\)C�Ф                                    By�?�  "          @'
=?�\)����?�Q�Aܣ�C�u�?�\)��z�?��A�p�C�Ф                                    By�Nj  	�          @(��@���?333Ax��C�� @���(�?
=AO�C�g�                                    By�]  �          @7
=@�\�޸R?fffA���C���@�\��?B�\Ay�C�B�                                    By�k�  T          @8Q�?Ǯ��33?�(�A�\)C��?Ǯ� ��?���A��C��f                                    By�z\  
�          @1G�?У׿�  ?�\)A�(�C��H?У׿���?�p�AӅC��3                                    By��  �          @6ff?��Ϳ�=q?�Q�A�C���?��Ϳ�Q�?���A�  C��3                                    By���  �          @@  ?���33?�
=BffC��?����
?��A�Q�C��R                                    By��N  "          @G
=?��R��?��HB�C��=?��R�\)?��A�C���                                    By���  T          @J=q?��
��?�z�A�33C���?��
�33?�p�A�C���                                    By�Ú  �          @P  ?�׿�ff?�
=B�C��?�׿���?��
B�C��
                                    By��@  �          @XQ�?�  ��ff@��B9�\C�<)?�  ��p�@z�B-ffC���                                    By���  "          @W�?��H��ff@%BI��C�W
?��H��  @�RB>Q�C�\)                                    By��  
�          @I��?����z�H@'�Ba  C���?�����
=@"�\BVffC�l�                                    By��2  
�          @.{?�ff�W
=?�G�B&ffC�y�?�ff�z�H?�Q�Bp�C�Ǯ                                    By��  _          @#�
?���?˅B
=C�
=?����?�ffBQ�C�]q                                    By�~  "          @4z�?�ff��R@��Bk�C�:�?�ff�O\)@ffBcp�C�#�                                    By�*$  I          @3�
?У׿z�@��BO
=C�&f?У׿@  @	��BH\)C��                                    By�8�  �          @:�H?u��@+�B��{C��\?u�:�H@(Q�B���C�l�                                    By�Gp  
�          @>�R?L�Ϳ!G�@333B��C��3?L�ͿW
=@/\)B�33C���                                    By�V  
�          @K�?333�8Q�@AG�B�.C��?333�s33@<��B�� C�33                                    By�d�  �          @N{?p�׾���@C33B���C�+�?p�׿\)@AG�B�33C�n                                    By�sb  "          @O\)?���@@��B�\C��3?���Q�@?\)B�L�C��f                                    By��  �          @L��?�
=��\@+�Bc�C���?�
=�5@(��B]��C�~�                                    By���  
�          @<��?�{�aG�@.{B���C�s3?�{��
=@,(�B�ffC��
                                    By��T  
�          @@  ?����
@.�RB�\C�Y�?���@,��B�#�C��                                    By���  
�          @<(�?�
=�
=q@!G�Bk=qC��
?�
=�8Q�@{BdQ�C���                                    By���  "          @;�?�\)��G�@,(�B�\)C�AH?�\)���R@*�HB���C�S3                                    By��F  T          @|(�?��
>�Q�@e�B��AQ?��
=�Q�@eB��@Q�                                    By���  
�          @�{?�ff?���@e�Bm
=Br{?�ff?�=q@l��B{��Ba�                                    By��  T          @�=q?W
=@��@�{Bt�B�L�?W
=?���@��\B�L�B��R                                    By��8  T          @�p�?J=q@�\@���Bq��B��
?J=q?�(�@�{B��)B�#�                                    By��  �          @���?p��?��R@q�BoG�B���?p��?ٙ�@z=qB~\)Bt��                                    By��  �          @���>�@+�@^�RBOB��)>�@�H@j�HB`Q�B��R                                    By�#*  {          @�Q��G�@k�@"�\B
p�B�ff��G�@^�R@333B=qB��{                                    By�1�  T          @1녾��@ ��?Tz�A�B�z���@��?�G�A��HB���                                    By�@v  �          @E��.{@+�?���Aԣ�B�33�.{@%�?��A�  B��                                     By�O  T          @U?��H?aG�@5Bl33A�\)?��H?+�@8��Bt
=A�(�                                    By�]�  �          @R�\?��H?�G�@Q�B@=qB]�
?��H?�=q@   BM�BRG�                                    By�lh  "          @mp�?���@�@   B+�BV�R?���?��R@*=qB8G�BL�
                                    By�{  �          @�G�?�z�@>{@'
=B�
Bs  ?�z�@1�@4z�B$(�Bl33                                    By���  �          @�Q�?˅@g
=@��A�B��=?˅@\(�@p�B�HB��                                     By��Z  �          @��?\@W�?��HA���B�G�?\@QG�?���A�\)B�
=                                    