CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240712000000_e20240712235959_p20240713021629_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-13T02:16:29.591Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-12T00:00:00.000Z   time_coverage_end         2024-07-12T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy
C�   
�          @�Q�?��
���\��z����
C���?��
������p���
=C���                                    By
C�  
(          @�
=?}p��A��B�\�.��C�?}p��>�R�E�2{C�(�                                    By
C�L  
�          @�=q?��Dz��J�H�/��C�g�?��AG��N{�3(�C���                                    By
D �  T          @�(�?�z��N�R�@  �!�C��f?�z��K��C33�%�C��                                    By
D�  T          @�  ?�\)�K��N�R�+�HC��=?�\)�H���Q��/
=C��{                                    By
D>  �          @�p�?��H�8���U��7{C�l�?��H�5�W��:�C��                                     By
D,�  "          @��
?�G��g
=��R��\C��)?�G��e���\���C�
                                    By
D;�  "          @��\?��
�k��G�����C��?��
�i�����ظRC��                                    By
DJ0  �          @���?�\�Y���Q���ffC���?�\�W���H�{C��R                                    By
DX�  T          @�G�?��
�9���<���#��C���?��
�7
=�?\)�&�C��                                    By
Dg|  �          @��R?�Q��{�P���=�C�/\?�Q����S33�@G�C�n                                    By
Dv"  
�          @�z�@=q�tzῂ�\�LQ�C�!H@=q�s33�����V�RC�.                                    By
D��  
�          @��@*=q�r�\�8Q��(�C���@*=q�r�\�k��333C��=                                    By
D�n  �          @��
@ ���(���{���C�w
@ ���
=q�����C���                                    By
D�  
�          @�Q�?�\)=��
��  ��@X��?�\)>\)��  �)@�\)                                    By
D��  
(          @���?�녾\)��G��3C�"�?�녽��
��G���C�E                                    By
D�`  
�          @��?���+���\)��C�B�?����R���Q�C��                                    By
D�  �          @��R@zᾀ  ���
�wp�C��=@z�L�����
�w�C��=                                    By
Dܬ  
�          @�{@�
=�����r�
C��{@�
=q�����sp�C��f                                    By
D�R  	�          @��@�R�Y�������t
=C��H@�R�O\)����t�C��                                    By
D��  �          @��@녿�=q�����j�RC�^�@녿��
����kC���                                    By
E�  
�          @���@(��p�����
�oC�W
@(��fff��(��p�C��3                                    By
ED  �          @��?�����y���g{C��{?���  �z�H�h�\C�3                                    By
E%�  �          @��H@ �׿�G��n{�YG�C�c�@ �׿�(��o\)�ZC���                                    By
E4�  
�          @��R?�G�����S33�S�HC�<)?�G��
=�Tz��UC�l�                                    By
EC6  T          @��\@��z��fff�cG�C�h�@����fff�c�C�Ф                                    By
EQ�  T          @�(�@\)�!G��l���g�C�.@\)�
=�mp��h=qC��{                                    By
E`�  
�          @��@�\�u�l(��h=qC�\@�\�L���l(��hp�C�u�                                    By
Eo(  �          @��H@,���?\)��
=��C�\@,���>�R��G�����C�3                                    By
E}�  "          @��@{�l��?�(�At��C��q@{�mp�?�Q�AnffC��{                                    By
E�t  �          @�@\���HQ�>\)?�G�C��@\���HQ�=�?��HC���                                    By
E�  �          @�\)@L(��]p��Ǯ��
=C�S3@L(��]p��������C�U�                                    By
E��  
Z          @��R@8Q��g��:�H�z�C�=q@8Q��g
=�B�\���C�B�                                    By
E�f  "          @��R@333�j=q�����G�C���@333�j=q����33C��\                                    By
E�  T          @�33@���s�
�!G���
=C�
=@���s33�&ff�z�C�                                    By
Eղ  T          @��@��tz�>�ff@���C�` @��u�>�(�@�  C�]q                                    By
E�X  �          @��\@Q��p  ?�  AJ=qC�7
@Q��p  ?xQ�AEC�1�                                    By
E��  "          @��@���q�?Tz�A*�\C�j=@���q�?O\)A&=qC�ff                                    By
F�  T          @��@33�U�?ٙ�A���C�˅@33�U?�
=A���C���                                    By
FJ  �          @r�\?�ff�'�@�\B�\C�E?�ff�(Q�@�B��C�8R                                    By
F�  T          @u�?����@!�B.=qC���?���(�@!G�B-p�C���                                    By
F-�  	�          @{�?h�ÿ�  @eB�p�C�\?h�ÿ�G�@e�B�\C��f                                    By
F<<  �          @��
>�Q�&ff@���B�G�C�o\>�Q�(��@���B��fC�(�                                    By
FJ�  
�          @l��?\)�
=q@]p�B��{C��?\)���@]p�B�Q�C��                                    By
FY�  "          @�ff?�G����vff�p�\C�p�?�G���ff�vff�q�C���                                    By
Fh.  T          @�33?�G����z=q�i��C�j=?�G��
�H�z�H�j33C�w
                                    By
Fv�  �          @�(�?�Q��
=�]p��Rz�C�XR?�Q��
=�^{�R��C�b�                                    By
F�z  
�          @qG�@   �(�?   A�HC��@   �(�>��HA��C���                                   By
F�   "          @��@ff�P  ���Ϳ�z�C�w
@ff�P  ���Ϳ�p�C�w
                                   By
F��  
�          @��?�  �W
=@��B���C���?�  �aG�@��B���C�ٚ                                    By
F�l  
�          @��@33��@S�
BS�C��@33��@S33BS�
C���                                   By
F�  T          @��@���Q�@S�
BK�C�k�@���Q�@S�
BK�HC�h�                                   By
Fθ  T          @�ff@{�@Z=qB:G�C��f@{�@Z=qB:G�C��f                                    By
F�^  
�          @�{@#33�1G�@2�\BQ�C�P�@#33�1G�@333B\)C�Q�                                    By
F�  
Z          @�z�@(Q��,��@,��Bp�C�q@(Q��,��@,��B�\C�!H                                    By
F��  "          @��R@.{�>�R@�A��
C�&f@.{�>�R@�A�=qC�(�                                    By
G	P  
�          @��@,���N{?�33A\C�  @,���N{?�z�A��C��                                    By
G�  T          @���@
�H�J=q�9���p�C�9�@
�H�J=q�8���{C�4{                                    By
G&�  
�          @��?�  �I���N�R�'z�C��?�  �J=q�N{�'
=C��f                                    By
G5B  
�          @�33?����C�
�`  �8��C��?����Dz��_\)�8ffC��                                    By
GC�  "          @�33?�z��C�
�U�/ffC�7
?�z��Dz��U��.C�,�                                    By
GR�  
�          @�z�?Ǯ�w�������C���?Ǯ�xQ��ff��(�C���                                    By
Ga4  "          @��H?�\�~�R������C�  ?�\�\)��\)���
C��)                                    By
Go�  "          @�  ?�(���z������C�8R?�(���z�����C�4{                                    By
G~�  �          @��\@#�
�|(���ff�J{C���@#�
�|(����
�FffC��                                    By
G�&  "          @�  @	���vff��
=��z�C���@	���w
=��z���ffC��=                                    By
G��  T          @���@33�p  ��{��\)C�@33�p�׿����C���                                    By
G�r  �          @���@33�w������aC�Y�@33�xQ쿎{�\��C�S3                                    By
G�  "          @��
?�p���
=��ff���\C�"�?�p���
=��
=���C�                                      By
GǾ  "          @�p�?�(���{��\�ǮC��
?�(���ff����z�C��{                                    By
G�d  
Z          @�=q@�\��녿����C��@�\��=q����G�C���                                    By
G�
  
�          @���?����ÿ�=q�TQ�C��=?����ÿ�ff�MC��                                    By
G�  "          @�@33���׿�G��IC���@33���ÿz�H�C33C���                                    By
HV  T          @��?�
=�8���L(��6C�  ?�
=�:=q�J�H�4��C��                                    By
H�  �          @��?k��������R�HC�]q?k���{��{��C�q                                    By
H�  
�          @��?!G���33���R�{C�%?!G���Q���{�\C���                                    By
H.H  
�          @��>���<���s33�O\)C��\>���?\)�qG��M(�C��H                                    By
H<�  
Z          @���?E��,(��u�W�RC�f?E��.�R�s�
�Up�C��                                    By
HK�  
�          @�(�?�33�Q������bp�C��?�33���\)�`=qC��3                                    By
HZ:  �          @��
?�ff�
�H�����jz�C�g�?�ff�{���
�hG�C�%                                    By
Hh�  
�          @�  ?z�H�xQ��,���
=C��?z�H�z=q�)���p�C�                                    By
Hw�  
Z          @��?��
�HQ��@���)�RC�{?��
�J�H�>{�'{C���                                    By
H�,  �          @�(�?���[��P  �,  C���?���^{�Mp��)(�C�޸                                    By
H��  "          @�Q�<��
���
��=q=qC�aH<��
������C�^�                                    By
H�x  T          @�(�?@  �:�H�n{�K�HC�4{?@  �>{�l(��H�C�
                                    By
H�  "          @��
>�ff�HQ��k��EG�C�
>�ff�L(��h���B(�C�f                                    By
H��  �          @���?��p������u=qC��?��G����
�r
=C���                                    By
H�j  	�          @�{?(�ÿL����z�\)C��3?(�ÿ^�R��(�
=C��)                                    By
H�  
�          @�z�?�p��P���:=q�{C�H?�p��S33�7
=���C�ٚ                                    By
H�  
�          @��
?����7��b�\�=  C�P�?����;��`  �9�
C��                                    By
H�\  	�          @��@33�L���:�H�(�C�XR@33�O\)�7��
=C�'�                                    By
I
  "          @�p�?��|(�� ����33C���?��~�R�(���  C�Ф                                    By
I�  "          @�
=@���
��=q�z=qC��@��zῡG��lz�C��q                                    By
I'N  "          @���@���n�R�   ��p�C��@���p�׿�
=��z�C��3                                    By
I5�  �          @���@
�H�|�Ϳfff�4��C�n@
�H�}p��Tz��&{C�aH                                    By
ID�  �          @��
@{�xQ�#�
�ffC�޸@{�x�ÿ���
=C���                                    By
IS@  
Z          @�ff?�(�����@��RB�G�C�R?�(��#�
@��RB�\)C��R                                    By
Ia�  
�          @��R?�=q�.{@��\B�Q�C�]q?�=q��\)@��\B�z�C��
                                    By
Ip�  
(          @��?�zῗ
=@�p�B�#�C�{?�zῊ=q@�ffB��{C�>�                                    By
I2  T          @���?��
��(�@�p�B���C�8R?��
��\)@�{B�33C�e                                    By
I��  �          @���>��þ��H@�=qB�#�C��>��þ\@��\B�{C�(�                                    By
I�~  �          @��H?��H��ff@�p�Bs�RC���?��H���H@��RBw�C�<)                                    By
I�$  
�          @��@@  ��\)@L��B,�RC��@@  ��ff@O\)B/{C�U�                                    By
I��  "          @�=q>�ff��=q@��
B�u�C�޸>�ff��p�@��B���C�>�                                    By
I�p  "          @�?��
���H@���Bl�HC��?��
��\)@�{Bpz�C��                                    By
I�  �          @�@����@��HBbffC�� @���Q�@��
Bd�C��=                                    By
I�  
�          @�G�@z��>�R?��A�z�C���@z��<(�?�z�A�33C��                                    By
I�b  T          @�p�@#33�.{@Mp�B"�C���@#33�(��@QG�B&��C��                                    By
J  "          @�ff@?\)��z�@U�B,z�C��R@?\)��=q@W�B/�C�L�                                    By
J�  
�          @�(�?�Q��W�@fffB4�HC�?�Q��Q�@l(�B:ffC��                                    By
J T  �          @��H>����@5B�\C�'�>����\@<��B�C�7
                                    By
J.�  T          @���?��n{@6ffB
ffC��{?��h��@<��B�HC�0�                                    By
J=�  "          @��?�p��S�
@I��B{C�e?�p��N�R@O\)B"ffC��)                                    By
JLF  T          @�{@�
�]p�@5B��C�k�@�
�XQ�@;�B
=C��
                                    By
JZ�  �          @��@���~{?�p�A��C���@���{�?˅A��\C���                                    By
Ji�  "          @�p�?����G�?.{Az�C��?������?O\)A!�C��                                    By
Jx8  �          @�{�L������?
=q@�ffC��{�L������?.{A{C��{                                    By
J��  
�          @�\)�z�H���׿u�)�C�� �z�H�����O\)�33C��f                                    By
J��  	.          @�(��G���33�"�\���C���G����=q��p�C���                                    By
J�*  T          @��H��H�s�
�33�ÅCm�=��H�w���
=����Cm�3                                    By
J��  T          @�  ���R����>\)?��
C}Ǯ���R����>�=q@`  C}�                                     By
J�v  �          @��?h����Q�@G�A���C�e?h����{@�HA�G�C�z�                                    By
J�  
�          @�G������?��@��C�W
����z�?B�\A�\C�T{                                    By
J��  "          @���������R?��A=G�C��3������p�?���A[\)C��\                                    By
J�h  �          @�  ?�z�ٙ�@H��BS�C�8R?�z�˅@L��BX�C�%                                    By
J�  T          @��@�
=L��@�
=BuG�?�(�@�
>B�\@��RBt�H@���                                    By
K
�  T          @�Q�?�녿��\@���Bz�
C�
=?�녿�{@�{B{C��3                                    By
KZ  "          @��@{���\@��\Bm�HC�  @{��\)@�(�Bq�C���                                    By
K(   "          @�{@�Ϳ��@���B_C�� @�Ϳ��@��\Bc��C�33                                    By
K6�  
�          @��@{����@���Bb��C�n@{��(�@�33Bfz�C���                                    By
KEL  "          @��@333�Q�@w�B>�HC�k�@333��p�@|��BC�HC�`                                     By
KS�  
�          @��\@B�\�G�@mp�B1z�C�� @B�\�Q�@r�\B6�C�}q                                    By
Kb�  T          @��\@W��&ff@K�B��C�(�@W���R@Q�B(�C��3                                    By
Kq>  
�          @�G�@o\)�%@)��A�=qC���@o\)��R@0  A�{C�7
                                    By
K�  "          @�G�@fff�A�@
=A�
=C���@fff�;�@{A�ffC�l�                                    By
K��  T          @��@j�H�z�@;�B(�C��
@j�H���@AG�B�C���                                    By
K�0  	�          @��H@dz��   @<��Bp�C�� @dz��Q�@C33B��C�(�                                    By
K��  T          @�G�@N�R�9��@@��B
�HC�\@N�R�1G�@HQ�B
=C��\                                    By
K�|  �          @��?���{?���A�p�C�q?����
?�G�A���C�C�                                    By
K�"  
�          @���?����(�>���@c�
C��q?�����?
=q@���C��                                    By
K��  "          @���?�p���ff?�33AK33C��3?�p�����?�{Ap��C�f                                    By
K�n  "          @��\?�33��  ?�  A[
=C�w
?�33��{?��HA��RC��=                                    By
K�  T          @�G�?�(���  ?��AG�
C��R?�(���{?�{An�RC��=                                    By
L�  
�          @�Q�?   ���R?�z�A|��C��)?   ��z�?У�A�z�C��f                                    By
L`  
�          @���?=p����
?��
Ah��C�W
?=p����?�  A���C�e                                    By
L!  
�          @�\)?�Q��,(�@�Q�B]�\C��?�Q��   @�(�Bg  C��{                                    By
L/�  "          @��R?�
=�.�R@���B\z�C��
?�
=�!�@�z�Bf{C��f                                    By
L>R  
�          @�{>�ff�J=q@��BO
=C��>�ff�>{@�ffBY�\C�S3                                    By
LL�  
(          @��?Y���:=q@�  BY��C�.?Y���-p�@�(�BdG�C�                                    By
L[�  T          @���>���9��@��Bb��C�xR>���,(�@���Bm�\C���                                    By
LjD  
�          @��?W
=�   @�z�B��C�ff?W
=��\@�\)B���C��                                    By
Lx�  
Z          @�ff?���@���Bk��C�H�?��У�@��BsQ�C�f                                    By
L��  �          @�{@+���@b�\B2�C�AH@+����@i��B9�
C�Z�                                    By
L�6  
(          @��\?��H��{@��B~�C��?��H��\)@�=qB�p�C��                                    By
L��  
�          @�33?�����@���B�.C�33?��p��@��\B�{C���                                    By
L��  
�          @�33?��H���@���B��C�b�?��H�aG�@��B�B�C��                                    By
L�(  T          @��
?�ff��ff@��HB��{C���?�ff�J=q@�z�B���C�q�                                    By
L��  "          @��H?�ff�(�@�B���C�f?�ff����@�ffB�
=C�0�                                    By
L�t  �          @�\)?�G�����@��B���C�9�?�G���@�=qB�Q�C�U�                                    By
L�  
�          @�G�?
=����@�B���C�8R?
=��{@�Q�B���C��
                                    By
L��  "          @�{?+��Q�@�33Bu�C���?+���33@��RB��qC��{                                    By
Mf  
�          @�\)?���Q�@\)Bk{C�)?�녿�z�@�33Bv  C�u�                                    By
M  T          @�ff?�Q쿅�@�=qB�G�C�7
?�Q�E�@��
B�G�C��H                                    By
M(�  	�          @���?��ÿ�@��HB���C��?��þu@��B�z�C��\                                    By
M7X  
�          @�Q�?�\)?���@�Q�B��=B/{?�\)?�{@�{B�ǮBI��                                    By
ME�  
�          @�
=?�  ?�{@�G�B�33B)G�?�  ?У�@�ffB��\B>{                                    By
MT�  "          @�  ?���>�ff@��
B�Q�AY?���?:�H@��HB�Q�A�{                                    By
McJ  T          @�(�?���?�=q@��\B�A�  ?���?���@�Q�By  B
z�                                    By
Mq�  
�          @�
=?�z��p�@�G�B��C���?�zῺ�H@�(�B�#�C�N                                    By
M��  �          @�G�?
=q�\��@|(�BA��C�w
?
=q�Mp�@�z�BOQ�C�˅                                    By
M�<  �          @���=�����G�@�RA��\C���=�����(�@/\)B�C���                                    By
M��  
�          @�  ?333��G�?�{A��C���?333���@�A�p�C��                                    By
M��  
�          @�Q�?8Q���G�?��\AR�RC���?8Q���
=?��
A�33C��q                                    By
M�.  
�          @��
>�����?���A��\C��)>�����R?�(�A�
=C��f                                    By
M��  �          @�z�@=p�?
=@�z�BW��A3
=@=p�?W
=@�33BTffA~�H                                    By
M�z  
Z          @���@W
=?�R@~{BE=qA&�R@W
=?^�R@{�BB{Ah                                      By
M�   "          @�@c�
=L��@n{B8?O\)@c�
>�z�@mp�B8{@�(�                                    By
M��  T          @��@P�׿c�
@vffBC�C�` @P�׿#�
@y��BFz�C�w
                                    By
Nl  
Z          @�=q@p��?#�
@n{B1G�A
=@p��?aG�@j�HB.Q�AS�
                                    By
N  �          @�=q@~�R=��
@aG�B%�?��@~�R>��R@`��B%(�@�
=                                    By
N!�  �          @��R@tz�>�@I��B{?��@tz�>���@H��B(�@�Q�                                    By
N0^  	�          @��@��;���@QG�B��C��{@��ͽ�G�@R�\Bp�C�9�                                    By
N?  �          @�{@�(����@L(�B�C�Z�@�(��B�\@Mp�B  C��f                                    By
NM�  �          @�  @����\)@7�B
=C�0�@�����
@8Q�B�C�z�                                    By
N\P  �          @���@�녿���@Q�A�
=C���@�녿�ff@�RAң�C��                                     By
Nj�  T          @�\)@�{�0��?�G�A6ffC�P�@�{�+�?���AYG�C��\                                    By
Ny�  
�          @��@R�\�c33?�R@陚C�aH@R�\�`  ?^�RA$(�C��R                                    By
N�B  
�          @�=q@�����?(��@�\)C�H@����{?uA4  C�,�                                    By
N��  	�          @�>.{��(��
=��=qC�H>.{�����   ��G�C��R                                    By
N��  	�          @�{?u��(�=��
?}p�C�/\?u���>Ǯ@�G�C�5�                                    By
N�4  
�          @�ff?���z�W
=�C�W
?�������{C�>�                                    By
N��  
�          @�ff?�\��G��5���HC���?�\���\��{�fffC��H                                    By
Nр  
Z          @��@%���R?���A��C��@%���\?�A�
=C�q�                                    By
N�&  T          @�Q�@@����
=?\(�AG�C��)@@����z�?�Q�AJ=qC�8R                                    By
N��  
�          @��@Q����
?E�A z�C�k�@Q�����?���A7�
C��                                    By
N�r  
(          @�
=@e��z�u�(�C�z�@e��(�>��@,(�C��H                                    By
O  "          @���@QG��?\)��=q���\C�Ǯ@QG��Fff�����z�C�@                                     By
O�  �          @��@y���aG��L���	C��@y���e������
C��)                                    By
O)d  
Z          @��@���Mp������z�C�p�@���O\)�����K�C�L�                                    By
O8
  
�          @��\@�Q��<�Ϳ��H�R�\C���@�Q��B�\�z�H�(��C�>�                                    By
OF�  
(          @�\)@�
=�E���ff���C���@�
=�L(���ff�\Q�C�y�                                    By
OUV  
Z          @��@��\�#33��  �UC�p�@��\�(�ÿ���1p�C�H                                    By
Oc�  �          @�ff@��R�L(���
=�Ip�C�s3@��R�QG��k��Q�C��                                    By
Or�  �          @��
@HQ���녿�p����
C�@HQ�����\)�b�HC��                                     By
O�H  T          @��\@S33���H��\)���HC�u�@S33��\)���
�~�RC��q                                    By
O��  
(          @�=q@Q��\)�z����HC��R@Q����Ϳ�p���G�C�,�                                    By
O��  T          @��@R�\�w��  ���C�1�@R�\����������C��3                                    By
O�:  
�          @���@i���S�
�\)�ָRC��@i���`���p���Q�C��                                    By
O��  �          @��@Tz��a��,(���=qC��H@Tz��o\)�Q���p�C���                                    By
Oʆ  
(          @�@1��������HC��@1���33�����p�C���                                    By
O�,  "          @��@G
=�������  C���@G
=���ÿٙ����C�:�                                    By
O��  
�          @��@Mp��o\)�   ���HC�N@Mp��|���
=q��=qC���                                    By
O�x  �          @�G�@AG��u�&ff���\C��@AG�����������\C�S3                                    By
P  �          @�ff@$z��~{�*=q��
=C�}q@$z���{�33����C�˅                                    By
P�  
�          @�@�H�x���9���33C��@�H��(��"�\��p�C�*=                                    By
P"j  
(          @��@���x���:=q�  C�ٚ@����(��#33��RC�{                                    By
P1  �          @�{@'
=�s33�8Q����C�@ @'
=�����!G���
=C�n                                    By
P?�  
�          @��@���~{�1���C���@����ff�=q��C���                                    By
PN\  "          @�  @���h���6ff��C���@���xQ��   ���C�#�                                    By
P]  �          @��@(Q��`���-p�����C�l�@(Q��p  �Q���G�C���                                    By
Pk�  T          @�33@�H�N�R�G����C�s3@�H�`  �333�Q�C�W
                                    By
PzN  
�          @�ff@  �7������?=qC�f@  �N�R�n�R�-��C�e                                    By
P��  "          @�z�?�����{�q�C�?�����  �b{C��                                    By
P��  
�          @�z�?�����H���\�|��C�Z�?�������p��n{C���                                    By
P�@  �          @�@*�H�y���J�H�\)C�33@*�H��{�1���{C�E                                    By
P��  �          @�=q@���C�
��Q��7�
C��R@���[��l���&
=C�ff                                    By
PÌ  �          @�G�@=q�J=q�w��0�HC��=@=q�aG��b�\���C�1�                                    By
P�2  "          @�G�@A��\)�o\)�.�\C�C�@A��6ff�^�R��RC�b�                                    By
P��  �          @�Q�@xQ�����w
=�2��C�
@xQ�E��s33�/G�C�]q                                    By
P�~  
Z          @��@^�R��(��q��4
=C���@^�R���g��*G�C��                                    By
P�$  �          @��@S�
��=q�q��4
=C��@S�
����e��(�C�1�                                    By
Q�  �          @��\@L(���p��y���<=qC���@L(����mp��0Q�C�5�                                    By
Qp  �          @��
@I���޸R�\)�?�C��f@I������r�\�3C��=                                    By
Q*  �          @���@Fff������Q��@\)C���@Fff�{�s33�3��C�:�                                    By
Q8�  �          @��@O\)��G��z�H�>�C���@O\)��33�p  �3�RC���                                    By
QGb  
Z          @���@q녿n{�n�R�.�HC�3@q녿���g
=�(�C�o\                                    By
QV  �          @��@c33��ff�r�\�6�C���@c33��Q��j=q�.��C���                                    By
Qd�  �          @��R@S�
����u�>z�C��@S�
���
�l���5�C��{                                    By
QsT  �          @�\)@S�
�����y���A=qC�f@S�
��p��qG��8�C��R                                    By
Q��  �          @��@b�\�����k��2�RC�/\@b�\��G��b�\�*=qC�u�                                    By
Q��  �          @�Q�@\(��z�H�vff�<��C�3@\(������n�R�4�C��                                    By
Q�F  �          @�  @Q녿fff��  �Fz�C�L�@Q녿�=q�xQ��>�\C���                                    By
Q��  "          @�p�@(������\���
C���@(���=q��G��U��C�8R                                    By
Q��  "          @��@����{�
=�̸RC��@����p���{��p�C�S3                                    By
Q�8  �          @�ff?�
=�`  �k��*z�C�l�?�
=�xQ��QG��33C�5�                                    By
Q��  T          @��?ٙ��&ff���\�Y33C���?ٙ��C�
��Q��C�C���                                    By
Q�  
Z          @�p�@33�L���j=q�+Q�C��)@33�e�Q��=qC�XR                                    By
Q�*  T          @�\)@J=q�n{�����p�C�0�@J=q�~{�G����
C�Ff                                    By
R�  
�          @�G�@?\)�i���0����=qC���@?\)�|(����ɮC���                                    By
Rv  T          @���@333�4z��tz��/ffC�]q@333�N�R�^{�\)C�j=                                    By
R#  �          @��
@)���HQ��u��,{C�q@)���c33�\�����C�aH                                    By
R1�  �          @�Q�@
=q�Vff�s33�.G�C�k�@
=q�p���X����
C���                                    By
R@h  �          @��@��G������PQ�C�u�@��0  �~�R�=Q�C��q                                    By
RO  
�          @�(�@\)���p��]33C�y�@\)�%�����J  C�k�                                    By
R]�  	�          @��R@�(���{�Lz�C�� @�:=q�xQ��8Q�C�^�                                    By
RlZ  �          @��H@0���J�H�j=q�$\)C���@0���dz��P����C��\                                    By
R{   �          @�Q�?�33�7
=��
=�K�\C���?�33�U�w
=�4p�C��f                                    By
R��  T          @���?:�H��\)����k�C���?:�H�{��z��y
=C�C�                                    By
R�L  �          @�=q?���
=q��  ���C�޸?���/\)���R�g�C�AH                                    By
R��  T          @��\?����z�������C�P�?����*=q��Q��k
=C��                                     By
R��  �          @��\?c�
�G����
aHC��?c�
�'����H�p�C�c�                                    By
R�>  �          @��H?G��G���(�8RC�z�?G��(Q���33�q��C�4{                                    By
R��  T          @��?k��   ��(��=C�E?k��'���33�q  C���                                    By
R�  �          @��?�\)�A�����Rz�C�{?�\)�c33���H�9(�C�@                                     By
R�0  �          @�(�?^�R��\)�J=q�
C�|)?^�R���H�%��ݮC�
                                    By
R��  T          @��H?Tz��^{����G�\C��\?Tz��~{�q��+�C�޸                                    By
S|  "          @�33?�=q�K���(��O  C�U�?�=q�mp��|(��433C�޸                                    By
S"  �          @��\?���9����Q��WC��)?���\������=��C��                                    By
S*�  T          @��H?��H�:�H��ff�SG�C�*=?��H�]p������9�C�"�                                    By
S9n  �          @��H?�
=�@�������H�C�P�?�
=�a��xQ��/C�O\                                    By
SH  
�          @��\@�
�'����R�U{C��@�
�J�H���H�=33C��                                    By
SV�  T          @��?���E��\)�K  C�C�?���g
=�s33�0(�C��3                                    By
Se`  
(          @�\)?�33�J=q��\)�JC��{?�33�k��q��/G�C�k�                                    By
St  T          @�\)?�\)�6ff�����R{C��\?�\)�X���xQ��7��C���                                    By
S��  
�          @�  ?��'
=���R�Z(�C�aH?��J�H���H�@�RC�޸                                    By
S�R  T          @�(�@7
=�ٙ������MQ�C���@7
=��R�x���<33C��q                                    By
S��  T          @��@Tz��ff�\)�=��C�|)@Tz���
�p  �/(�C�3                                    By
S��  T          @��@N{��=q����B�C���@N{�
=�s�
�2��C�e                                    By
S�D  T          @��R@Q녿�G��\)�;�HC�޸@Q����n{�+��C��R                                    By
S��  �          @�{@mp������n�R�-��C�@mp����aG��!z�C��{                                    By
Sڐ  T          @��R@P  �Ǯ���H�BG�C�+�@P  �ff�u�3{C��3                                    By
S�6  �          @�@@�׿�����E�C�H�@@���Q��u��3��C��{                                    By
S��  "          @�p�@C33���������@�HC���@C33��R�p  �.z�C�p�                                    By
T�  �          @�{@6ff����
=�KQ�C��
@6ff��R�z=q�8
=C�p�                                    By
T(  T          @�z�@8�ÿ��
��  �Q��C�{@8���ff��  �@��C�f                                    By
T#�  �          @�@?\)���H�����Pz�C��R@?\)��\�����@ffC��R                                    By
T2t  �          @�@*�H�ff���\�H��C��f@*�H�)���p  �3�\C���                                    By
TA  �          @�{?����(Q���Q��b��C��?����N�R��33�E�\C���                                    By
TO�  �          @�{?J=q�J�H����Sz�C��
?J=q�p  �u��3��C��                                    By
T^f  
�          @�
=?(���c33���H�B33C�J=?(�����H�c33�"(�C���                                    By
Tm  �          @��R@Q������hffC��@Q��#33��G��P�
C��                                    By
T{�  �          @�?��H�33���
�i�C��)?��H�,(������Pp�C�                                    By
T�X  
�          @�z�?
=q�<����ff�_��C�8R?
=q�c�
�~�R�?p�C�XR                                    By
T��  
�          @��?����7���p��\�C�y�?����^�R�}p��=\)C��                                    By
T��  �          @�?xQ��E����U��C��{?xQ��l(��w
=�5��C�\)                                    By
T�J  T          @�?@  �C�
����Z{C��H?@  �j�H�z�H�9ffC��                                    By
T��  �          @���?}p��{��p��q�C���?}p��G���Q��Q��C�Ф                                    By
TӖ  T          @��H?\(��C�
��
=�T33C��3?\(��h���n�R�3z�C��)                                    By
T�<  	�          @�{>�33���\����RC�]q>�33��z��2�\��C�!H                                    By
T��  
�          @���#�
�����1G���  C��q�#�
��(��z���ffC�G�                                    By
T��  �          @�=q��ff��\)�)����C�j=��ff���\�������\C�޸                                    By
U.  �          @�G����\)�X��� p�C��
����ff�0  ��ffC��                                    By
U�  T          @�?L���]p�����CG�C���?L�������_\)�!p�C��                                    By
U+z  �          @�
=?n{�L����=q�Q\)C��?n{�tz��q��/�RC�Ф                                    By
U:   
�          @��R?���\����p��HG�C��=?����G��fff�%�RC��                                    By
UH�  �          @���?B�\�i���x���8�HC��f?B�\��ff�R�\�ffC�%                                    By
UWl  �          @��H?W
=�]p��~�R�@�C�Ф?W
=�����Y���{C��                                    By
Uf  �          @��\?u�333���a{C��H?u�\���|���?33C��\                                    By
Ut�  �          @�z�?���(����\�f��C�o\?���5��|(��I33C��                                    By
U�^  "          @�  ��  �HQ��\)��HC��q��  �^�R��p���ffC��R                                    By
U�  
�          @��H�z���
=�aG��)�Cs�{�z������W
=�!�CtJ=                                    By
U��  "          @�z��p����Ϳk��0(�Cx�\��p�����aG��%Cx�3                                    By
U�P  
�          @��>�ff�\(��9����RC��3>�ff�vff�z���p�C�N                                    By
U��  "          @���?��Ϳ�Q�����\C��
?�����������h  C�S3                                    By
U̜  T          @���?˅�<���q��A��C�.?˅�`���P��� ��C�0�                                    By
U�B  T          @�G�@�R�p��r�\�C(�C�q@�R�A��Vff�&��C�+�                                    By
U��  "          @�
=?��R�"�\��  �V�\C�*=?��R�I���b�\�5�
C���                                    By
U��  
�          @��?���z���G��]�C�Ǯ?���<(��g
=�=�C��                                    By
V4  
�          @�{?�
=�
�H���H�`�
C��?�
=�333�l(��B  C�xR                                    By
V�  T          @�  ?�  �   �����X(�C�|)?�  �G��c�
�7(�C��
                                    By
V$�  T          @��\?��8Q��u��CC�{?��^{�S33�"�C���                                    By
V3&  
�          @��@���G��s�
�D33C�Ff@���7��X���(��C���                                    By
VA�  �          @�
=@'���=q�s�
�H  C��
@'����]p��0(�C��)                                    By
VPr  �          @�@G����|���T(�C�� @G��"�\�dz��9z�C��
                                    By
V_  �          @���@�ÿ�(��y���S��C�#�@����c�
�:�C��=                                    By
Vm�  T          @�33@0  ���tz��OC�t{@0  ����dz��=�\C�q�                                    By
V|d  T          @��?��
��p��\)�m��C�u�?��
�'��fff�L  C��                                    By
V�
  �          @�
=?���
=�����nC�7
?���333�w
=�LC�˅                                    By
V��  T          @�\)?��Ϳ�  ������C�?�����R����cp�C��f                                    By
V�V  
�          @���?��R�p�����c�C�  ?��R�8Q��j�H�B=qC���                                    By
V��  
�          @��@#�
�5��Z�H�'\)C�@#�
�XQ��8Q��	{C���                                    By
VŢ  T          @�z�@)���J�H����w
=C��3@)���Tz�z�����C�N                                    By
V�H  "          @�@.{�W
=?�
=A���C�}q@.{�Dz�?��RA�
=C��=                                    By
V��  "          @��H@U��Vff?W
=A!C�l�@U��I��?�z�A�z�C�H�                                    By
V�  T          @���@B�\�_\)?�p�An{C���@B�\�N{?�A�=qC���                                    By
W :  
�          @���@1G��e?�z�A�C���@1G��R�\@ ��A�=qC�
=                                    By
W�  
�          @�\)@G��O\)?�ffA�33C��)@G��=p�?���A��C�E                                    By
W�  �          @���?�z�����?�
=A�33C�  ?�z��|(�@=qA�z�C���                                   By
W,,  T          @��@
=��\)?���A}C��
@
=�z�H@ffA��C���                                   By
W:�  T          @�G�@;��l��?�\)AUC�8R@;��\(�?�G�A��
C�:�                                    By
WIx  T          @�@c�
�g
=?uA+\)C�Ff@c�
�X��?�=qA�
=C�4{                                    By
WX  �          @�
=@u��]p�?#�
@�33C��3@u��R�\?�  A^=qC��=                                    By
Wf�  �          @��@~{�W�>�@���C�� @~{�N�R?�ffA9C�t{                                    By
Wuj  
�          @�\)@�{�HQ�k��   C��R@�{�G�>�{@l��C���                                    By
W�  �          @��@l(��g
=>�=q@<(�C��@l(��`��?h��A"ffC�<)                                    By
W��  �          @��@z�H�U�L�Ϳ�C�˅@z�H�R�\?��@�p�C��q                                    By
W�\  �          @��@w
=�W
=����z�C�y�@w
=�Y��=�Q�?��\C�N                                    By
W�  T          @�{@�=q�L(��������RC��@�=q�Mp�>B�\@33C��{                                    By
W��  �          @�z�@z=q�@  ��z���C�@ @z=q�L�ͿW
=�ffC�Q�                                    By
W�N  
�          @�@h���E��������C��@h���XQ쿬���r�HC��\                                    By
W��  "          @�
=@g��   �1G�� ��C��\@g��>{�����  C�T{                                    By
W�  �          @���@z=q�.�R��ff��p�C���@z=q�AG����\�eC�0�                                    By
W�@  �          @��@]p��j�H��  �2�RC��=@]p��r�\�����S�
C�33                                    By
X�  T          @��@O\)�\)��\)�Dz�C���@O\)�~�R>��@���C���                                    By
X�  �          @�p�@c33�e�xQ��.{C�W
@c33�l�;�\)�H��C��H                                    By
X%2  T          @�(�@\(��p�׾�33�y��C�=q@\(��p��>�p�@�33C�>�                                    By
X3�  T          @��@Z�H�p�׿W
=���C�!H@Z�H�vff��G���G�C�˅                                    By
XB~  �          @�G�@�G��Tz�W
=�p�C�G�@�G��Z�H�B�\�C�޸                                    By
XQ$  �          @��@\)�S�
�.{��\)C�(�@\)�XQ�#�
��(�C�޸                                    By
X_�  �          @��R@{��Y������9��C��3@{��XQ�>Ǯ@�G�C��H                                    By
Xnp  �          @�Q�@y���`  �#�
��  C�f@y���^{?�\@��
C�*=                                    By
X}  T          @�  @��H�N�R����ҏ\C��H@��H�Q�<�>��
C��                                    By
X��  �          @��R@�ff�0  ��  �_�C�k�@�ff�<(��333���C���                                    By
X�b  
�          @���@�z���\)��  C��)@�z��!���\��z�C�G�                                    By
X�  �          @���@z=q���3�
�(�C��)@z=q�(Q����C��                                    By
X��  �          @�  @s33���
�G���\C�p�@s33�
=�.{��(�C��                                    By
X�T  �          @�
=@p  �����XQ��!ffC�J=@p  ���H�C33�\)C�9�                                    By
X��  T          @�33@g��5�^�R�-\)C�w
@g������QG�� �
C��
                                    By
X�  
�          @��\@n{���R�N{�C��@n{�����:�H�Q�C���                                    By
X�F  	�          @���@e����H�S33�$z�C���@e����@  ���C�e                                    By
Y �  
�          @�ff@n{���N{�"z�C��@n{��33�C33�z�C�j=                                    By
Y�  T          @�p�@[���(��[��3G�C�l�@[������Q��)G�C�3                                    By
Y8  
�          @�z�@E��2�\�!G����C��@E��N�R��z����C�Ф                                    By
Y,�  "          @�33@C33�(���\)���C���@C33�E���z���z�C�W
                                    By
Y;�  �          @�33?�G���p��ff���
C�g�?�G���Q쿜(��i��C��                                    By
YJ*  T          @��
?�G��l���7��{C�g�?�G���ff��
�ɅC�Y�                                    By
YX�  �          @�
=?�ff�:=q�AG��1{C��=?�ff�]p����C�p�                                    By
Ygv  �          @��R�\)�����n{�9��C���\)��z�<#�
=���C�q                                    By
Yv  
�          @��\>Ǯ��  �������C�|)>Ǯ��
=�
=��(�C�^�                                    By
Y��  T          @��
>\�S33�+����C�B�>\�q녿����مC�ٚ                                    By
Y�h  T          @�33?��R�\� ����RC�  ?��n�R�����z�C��
                                    By
Y�  T          @�ff@.{��(��+��Q�C��@.{��R�{���\C��{                                    By
Y��  "          @���@^{�  ��Q����C��@^{�#33��Q��v�HC��R                                    By
Y�Z  T          @�z�@@����׿�G���=qC���@@���%���  ��=qC��3                                    By
Y�   �          @�Q�@5��333��Q���(�C���@5��H�ÿ������RC��                                    By
Yܦ  �          @�33@u��
=q��{��{C�O\@u���ÿc�
�1C�                                      By
Y�L  T          @�33@b�\�!녿������C�1�@b�\�1녿c�
�2�HC��=                                    By
Y��  
�          @��@K��9����p����C��
@K��H�ÿW
=�*ffC���                                    By
Z�  �          @�G�@X���%��G����
C�L�@X���6ff�n{�=�C��
                                    By
Z>  �          @��@I���(Q��z���  C��@I���>�R�������C�L�                                    By
Z%�  T          @���@3�
�'�������C�~�@3�
�Dz��\��{C�C�                                    By
Z4�  �          @�Q�@�
����<(��$�C���@�
�AG��
=��ffC���                                    By
ZC0  T          @��?޸R� ���O\)�:�
C�Y�?޸R�H���(Q��  C�y�                                    By
ZQ�  "          @���?����{�H���=�
C��f?����5��&ff�
=C���                                    By
Z`|  "          @��@"�\���
�=p��.�C�s3@"�\�Q�� ���Q�C�p�                                    By
Zo"  
�          @��@XQ쾮{��R��\C�(�@XQ�\(��ff��C���                                    By
Z}�  
�          @��
@k��   �  ���
C�&f@k��xQ����Q�C��q                                    By
Z�n  �          @�p�@p  �c�
��㙚C�XR@p  ���ÿ���\)C�Q�                                    By
Z�  �          @�ff@333��\)�/\)��C�
@333��H�G���Q�C���                                    By
Z��  �          @��@J�H��G��  ���RC��@J�H��Ϳ�����p�C��q                                    By
Z�`  
�          @���?�ff�4z��5��%z�C�g�?�ff�W��	�����HC�Z�                                    By
Z�  �          @�  ?�G��p��Q��L(�C�%?�G��G
=�*�H��C��{                                    By
Zլ  
�          @��?s33����mp��q33C�h�?s33�)���L���B�\C���                                    By
Z�R  �          @��\?0������S33�_  C��?0���3�
�0  �.{C��                                    By
Z��  �          @���?�ff��p��b�\�{z�C���?�ff�{�HQ��OQ�C���                                    By
[�  "          @�Q�?�\)��\)�^{�q�\C��f?�\)��{�H���O�C��                                    By
[D  �          @J=q�L����
������C�XR�L���*=q���\��C�n                                    By
[�  T          @\)�!G��{�^�R��(�C~5ÿ!G��
=��Q���C�                                    By
[-�  "          @�녿E��9���?\)�3��C��E��_\)�G���C��f                                    By
[<6  �          @�
=�(���L���Q��4p�C�&f�(���vff��R���C��                                    By
[J�  �          @�=q�����N{�B�\�'33C{������s�
�\)��p�C}�f                                    By
[Y�  �          @�ff��\)�U��^{�2��C{n��\)�����(Q�� �
C~xR                                    By
[h(  �          @�  ����C33�^{�<{CzaH����p  �,(��
ffC}ٚ                                    By
[v�  
�          @�  ����N�R�K��)�Cw�ÿ���w
=�
=���HC{O\                                    By
[�t  
�          @����X���U�+�Cz��������R��G�C}��                                    By
[�  �          @�ff���H�X���J=q�  Cs=q���H�����33��z�Cv��                                    By
[��  
�          @�ff����Vff�8���
=Cl�����z�H�33���
Cp��                                    By
[�f  �          @�
=�z���z��(Q�� z�C��q�z���z��{��{C�h�                                    By
[�  "          @�녾����(��������C�t{������R�Tz���C���                                    By
[β  �          @�G��+���������z�C��=�+���z�n{�*=qC�{                                    By
[�X  �          @�
=�Ǯ�z�H�"�\����CxE�Ǯ���Ϳ������RCzs3                                    By
[��  
�          @�  �У��������  Cx�H�У�������z��W�CzQ�                                    By
[��  "          @�33���R�hQ��G��{Cw�3���R��  �����Cz��                                    By
\	J  
�          @�����Q������"�\��33CzO\��Q���Q���
��G�C|E                                    By
\�  
�          @�33�����ff������\C{�)������׿Y���#�C|ٚ                                    By
\&�  /          @��\����~�R�=p���\C~�R�����=q������Q�C�L�                                    By
\5<  T          @���+������?\)��\C�b��+���\)��
=���C���                                    By
\C�  "          @���?��������{�c��C��H?����S33�`���2ffC��                                    By
\R�  "          @��=�\)�h���`���/�HC��{=�\)����#33����C�|)                                    By
\a.  �          @�Q�������<(��z�C�&f����33��33��C�AH                                    By
\o�  
�          @��=������(���\)C��f=������{��C��3                                    By
\~z  �          @��׾�{�{��E�
=C��=��{������
�\C�޸                                    By
\�   �          @�G��   �tz��O\)� 33C�@ �   ��\)�\)��
=C���                                    By
\��  
�          @�zᾮ{�g
=�i���4��C�Q쾮{��(��+���G�C��=                                    By
\�l  �          @��þ\�i���^�R�-�C�׾\��(�� ���홚C���                                    By
\�  "          @�>�\)��H�����q(�C�U�>�\)�W��c�
�9�C�e                                    By
\Ǹ  �          @�(�>\�(���
=�t�\C�aH>\�\(��p  �=ffC�q                                    By
\�^  "          @��;���8Q���z��[��C�������r�\�Tz��$=qC��                                    By
\�  �          @�=q�W
=�W��i���9�C�R�W
=����.�R�  C�C�                                    By
\�  �          @�Q�u�H���vff�J�RC��f�u�~�R�>{���C�=q                                    By
]P  T          @��Ϳ#�
�=p���z��W\)C��#�
�xQ��S33� �C�XR                                    By
]�  �          @��R�#�
�e�o\)�8�C��3�#�
��z��0  � G�C��                                    By
]�  �          @�ff=L���c�
�p���:=qC�k�=L�����
�1����C�XR                                    By
].B  a          @���#�
�W
=�tz��BC��{�#�
��{�8���
(�C��
                                    By
]<�  �          @�z�#�
�A���z��W{C��R�#�
�}p��QG��\)C���                                    By
]K�  T          @�=q���:�H���
�Z�RC������u�Q��!�C�H                                    By
]Z4  "          @��þ��R�Dz��y���N\)C�����R�|(��AG���\C���                                    By
]h�  �          @��������g
=�a��1ffC�/\�������
�!���z�C�T{                                    By
]w�  
Z          @��
���G��W
=�<=qC�Ǯ���w
=��R�  C��                                    By
]�&  �          @��\�8Q��8Q��s33�OC�H�8Q��o\)�=p���C��                                    By
]��  �          @��ÿ   �\(��g��8��C��{�   ��\)�*=q��p�C���                                    By
]�r  T          @��þ��C�
�~{�P�C�����}p��E���C���                                    By
]�  �          @�33�z��J=q�|���K�\C��R�z������B�\�z�C��{                                    By
]��  
�          @��
�(��K��}p��JC����(���=q�B�\���C��{                                    By
]�d  
�          @��\��G��7
=��=q�U=qCz� ��G��r�\�N{��C
=                                    By
]�
  �          @��\��G��'���\)�a�RCx�\��G��e�[��*ffC~@                                     By
]�  
(          @�33���
�����o��Cv=q���
�W��k��8C}�                                    By
]�V  T          @�33�s33�����q�Cw�ÿs33�W��l���:G�C~33                                    By
^	�  �          @��H�s33�=q���
�m�CxY��s33�[��g��5�
C~u�                                    By
^�  �          @�녿E���\��ff�v�C{h��E��U�n{�=z�C���                                    By
^'H  T          @��R�!G��p���(��y=qC~(��!G��P���k��?�HC��=                                    By
^5�  "          @��R�^�R��p����RQ�CvQ�^�R�C33�s�
�H�C~�                                    By
^D�  �          @�\)�����������u�\ChT{�����;��o\)�C{Cs��                                    By
^S:  T          @�  �\��\)��{�x\)Cb�=�\�(���h���H\)Cp�                                    By
^a�  T          @�33��  ��\)���
�h\)Cb���  �7
=�`  �8�RCn��                                    By
^p�  �          @�z�\)�33�k��s{C�f�\)�;��A��7��C��q                                    By
^,  T          @�z�@G�����S33�8��C��@G��N{�#�
�{C��                                    By
^��  �          @�Q�?�z��2�\�G��/z�C�\)?�z��`  ��\���C���                                    By
^�x  T          @��R?����4z��I���4(�C��?����b�\��
��Q�C���                                    By
^�  T          @�  ?�G��'��Z=q�F33C��H?�G��Z=q�'
=��
C�(�                                    By
^��  �          @��?E��2�\�Z=q�F��C��H?E��e��$z��Q�C��                                    By
^�j  �          @���?aG��3�
�]p��F��C��q?aG��g
=�'
=�z�C��                                     By
^�  �          @�G�>�  �.{�g��S��C���>�  �dz��1��\)C���                                    By
^�  "          @��\?=p�����[��W  C���?=p��N{�+��p�C�xR                                    By
^�\  �          @��@G��=q�7��)p�C��@G��Dz������G�C��\                                    By
_  �          @��
@:=q������33C�
@:=q�2�\��z����\C��                                    By
_�  �          @���@ff�/\)�!G���RC�W
@ff�S33���H���C���                                    By
_ N  �          @�  @A��0  �������C�� @A��H�ÿ��
�U�C��                                    By
_.�  �          @�{@J�H�9�������  C��@J�H�R�\����Pz�C��\                                    By
_=�  T          @��@XQ��0  ������33C�o\@XQ��H�ÿ��
�H��C��
                                    By
_L@  �          @�{@a��%��ٙ����
C��f@a��<(��s33�;�C��                                    By
_Z�  �          @��@i������{��=qC�1�@i���,�Ϳ+���\C��                                     By
_i�  
�          @��
@j=q�(���33��
=C�"�@j=q�.�R�333�\)C��f                                    By
_x2  �          @��H@n�R��H����\z�C��H@n�R�'�������HC�o\                                    By
_��  "          @���@qG���
�c�
�5�C�B�@qG��p��u�C�
C�l�                                    By
_�~  �          @�p�@|(����   �θRC��@|(���p�<�>�ffC��f                                    By
_�$  �          @�33@Z�H�!G��Tz��0��C���@Z�H�*=q����33C�R                                    By
_��  �          @�{@U�\)�G��.�RC��@U�
=�����
C�^�                                    By
_�p  �          @\��@N{��G�?�AffC�O\@N{�Q�?Q�A^�RC��                                    By
_�  �          @s�
@g���?��\AyC��f@g���=q?��A�p�C��R                                    By
_޼  �          @e@N�R�u?�G�Aȣ�C��H@N�R>8Q�?\A��
@J=q                                    By
_�b  �          @mp�@C�
>�=q@33B�H@��@C�
?O\)?�33A��Am��                                    By
_�  
�          @s33@W
=���?�
=A�  C��q@W
=>��R?�33A���@��                                    By
`
�  �          @l(�@Z�H��ff?��A�ffC�9�@Z�H���
?���A�=qC�S3                                    By
`T  �          @\(�@HQ���?�{A��C�N@HQ켣�
?�A£�C�˅                                    By
`'�  �          @j�H@S33�k�?�=qA�Q�C���@S33>W
=?�=qA���@g
=                                    By
`6�  
�          @^�R@@  ���
?޸RA�33C�Ǯ@@  >�ff?�
=A�(�A	��                                    By
`EF  �          @w�@Mp�>W
=@
=qB�@n�R@Mp�?J=q@G�A�p�A]                                    By
`S�  �          @��R@_\)>���@B
=@��R@_\)?n{@
�HA��
Ao\)                                    By
`b�  �          @�  @]p�>���@�HB\)@�Q�@]p�?}p�@\)A��HA~=q                                    By
`q8  
�          @p  @J�H>�  ?��HA��\@���@J�H?G�?�A�  A\��                                    By
`�  
�          @\(�@E>Ǯ?��HAȏ\@�\)@E?G�?��A��RAb{                                    By
`��  T          @c33@C33?�?�p�A�(�Aff@C33?z�H?\A˅A�                                      By
`�*  �          @w�@7�?!G�@!�B"��AE�@7�?�ff@��BQ�A�\)                                    By
`��  �          @aG�@+�?J=q@Q�B33A�@+�?���?�=qA��RA��                                    By
`�v  T          @~�R@C33?p��@�B(�A�G�@C33?Ǯ@ ��A�p�AظR                                    By
`�  "          @k�@8Q�?�G�?��A�{A�\)@8Q�?޸R?�(�A�Q�A��H                                    By
`��  �          @���@A�?�Q�@#�
B��A�\)@A�?���@Q�A�G�A��                                    By
`�h  �          @��\@B�\?�33@%�B�HA㙚@B�\@33@G�A�33Bp�                                    By
`�  
�          @x��@4z�?���@G�B 33A�\)@4z�@z�?�  A�Bff                                    By
a�  "          @Q�@   ?�\)?�{A�z�B@   @ ��?��A�(�B�
                                    By
aZ  �          @�ff@u?�G�@$z�A�
=Aģ�@u@=q?�p�A��HB Q�                                    By
a!   "          @�p�@tz�?�33@'�B{A��H@tz�@z�@�
A�  A�(�                                    By
a/�  �          @��\@hQ�?�=q@B�\B  Aծ@hQ�@'
=@��A��B��                                    By
a>L  �          @�z�@]p�?�Q�@HQ�BG�A�
=@]p�@G�@&ffB p�B                                    By
aL�  �          @��R@O\)?�z�@XQ�B+��A��@O\)@#33@1�B  B�                                    By
a[�  �          @�
=@HQ�?��@g�B;��A��@HQ�@��@G
=B=qB�                                    By
aj>  �          @���@0�׾8Q�@w�BY�C�)@0��?k�@p��BQ�A�
=                                    By
ax�  �          @�\)@0  ��G�@s33BXp�C��)@0  ?xQ�@k�BN\)A���                                    By
a��  �          @���@<��>u@vffBQ�\@��
@<��?���@g�B@�A�z�                                    By
a�0  T          @�  @P  ?@  @mp�B@  APQ�@P  ?�\@UB(33A���                                    By
a��  �          @���@dz�?��@_\)B0
=A33@dz�?\@L(�B�RA�                                      By
a�|  
�          @���@s�
?�@N{B�\A33@s�
?��H@:�HB�\A���                                    By
a�"  �          @���@���>��@<��B
=@�(�@���?��@,(�B�HA��                                    By
a��  
�          @��R@�33?&ff@,��B�Az�@�33?�33@��A�  A�Q�                                    By
a�n  �          @��@��
?��@{A�A��@��
?��
@��A�  A�z�                                    By
a�  "          @�z�@W�@��?fffAK\)B��@W�@�
>�  @_\)B	                                    By
a��  
�          @�\)@Tz��G�@S33B3=qC�@Tz�?W
=@L��B,
=Abff                                    By
b`  �          @��
@Mp�>Ǯ@Q�B5@�p�@Mp�?���@A�B$�A�Q�                                    By
b  �          @�ff@B�\>�ff@c33BDffA
=@B�\?���@QG�B0�A�Q�                                    By
b(�  �          @��H@.�R>B�\@j�HBU(�@��\@.�R?��R@\��BDG�A��                                    By
b7R  T          @���@,(��\@y��B\��C��{@,(�?B�\@vffBW��A|(�                                    By
bE�  T          @��H@B�\�.{@Y��B@�C�o\@B�\?Q�@S33B9\)Ar{                                    By
bT�  T          @��@O\)�\@6ffB$��C��=@O\)>�@5B#�RA                                    By
bcD  T          @��@>{?�Q�@(�B33A��R@>{?���?��RA�
=A���                                    By
bq�  �          @�p�@fff�:�H?�G�A�=qC�>�@fff�B�\?��A݅C�xR                                    By
b��  �          @�(�@`  �   ?�(�A�
=C��@`  ��@�RA�z�C���                                    By
b�6  �          @��@Tz��p�@
=qA�C�=q@Tzῃ�
@$z�BC�e                                    By
b��  T          @�
=@8Q�=p�@=p�B3�\C���@8Q�>#�
@C33B:Q�@G
=                                    By
b��  T          @�\)@1G�����@J=qBBG�C��f@1G�?(�@G�B>�
AF�R                                    By
b�(  "          @��H@��.{@p��Bt{C���@�?n{@i��Bh�A��                                    By
b��  "          @�  @����H@�G�Bl(�C�5�@�?8Q�@�Q�Bh��A�(�                                    By
b�t  
          @�=q@�׿0��@�p�Bq�C��H@��?\)@�{Bs�
A_33                                    By
b�  
          @��
?�ff�n{@��B��C�W
?�ff>���@��B�W
AJ�\                                    By
b��  �          @��
?�33�u@�
=B��
C��?�33>���@��B�8RAX                                      By
cf  
�          @��H?W
=�^�R@��B�ǮC��R?W
=?�@��RB��B33                                    By
c  a          @��?\)��Q�@��B���C�.?\)?���@�(�B�� B�z�                                    By
c!�  �          @�{?�G�>��
@�ffB�� Ac�
?�G�?�{@��B(�BPz�                                    By
c0X  �          @�\)?�z���@��RB�
=C�Ǯ?�z�?J=q@��B�z�A�{                                    By
c>�  "          @�z�?��þ\@���B�Q�C���?���?\(�@��\B��Bz�                                    By
cM�  T          @�(�?�p����
@��B�z�C���?�p�?���@{�B�k�B�                                    By
c\J  "          @��
?�=q�B�\@vffB��\C��\?�=q?u@o\)B���B(�                                    By
cj�  "          @dz�?�{��p�@HQ�BxC�~�?�{?�@FffBt��A��R                                    By
cy�  T          @_\)@��aG�@��B>z�C�#�@�?   @
=B:  AP(�                                    By
c�<  
(          @x��@'
=���R@333B;
=C���@'
=?
=q@0��B7��A:�R                                    By
c��  "          @`  @:=q��?˅A�
=C���@:=q���
?�
=A�  C�Ǯ                                    By
c��  
(          @S�
@I���^�R����C�G�@I���^�R>�@�
C�G�                                    By
c�.  T          @E@2�\�G��p����Q�C�1�@2�\���
�+��J�RC��=                                    By
c��  �          @@  @p��8Q��G���z�C�˅@p���녿�p�����C���                                    By
c�z  �          @_\)@.�R�����˅���C��@.�R��{��=q���RC��H                                    By
c�   
�          @Vff@&ff��z��ff��C��@&ff�������(�C���                                    By
c��  �          @r�\@$z��\� ���.��C�aH@$z῞�R�\)��\C�#�                                    By
c�l  "          @�=q@.�R��R�N{�C��C���@.�R�Ǯ�8���*33C�%                                    By
d  T          @�  @+�>W
=�8���<33@�=q@+��(���4z��6ffC�\                                    By
d�  
�          @�G�@�\?��9���8
=B-�@�\?u�U�`33Aʣ�                                    By
d)^  
�          @���@%=��
�Vff�Q(�?��H@%�n{�N{�E�HC��                                    By
d8  
�          @�{@z��\)�Q��>��C��\@z��1G��$z��p�C��R                                    By
dF�  "          @��H?�p����H�e��_33C���?�p��-p��9���((�C�L�                                    By
dUP  �          @�ff@�
��z��O\)�MQ�C���@�
���.�R�%(�C�                                      By
dc�  �          @�Q�@��У��;��@{C��R@��(���
�  C�:�                                    By
dr�  T          @��?˅��\�J=q�K  C��\?˅�9�������
C�U�                                    By
d�B  "          @��R?����4z��0ffC��?��;�����{C��q                                    By
d��  T          @��@.�R���3�
�"33C�f@.�R�&ff�Q���G�C�,�                                    By
d��  
�          @��@*�H��Q��5�"��C��@*�H�-p�������C�L�                                    By
d�4  T          @��@�Ϳ�=q�AG��2��C��H@���*=q�z��
=C�Q�                                    By
d��  "          @�33@Q��G��!G��(�C�5�@Q��;���(����
C���                                    By
dʀ  T          @���@!G����Y���_33C���@!G���R���Ϳ�Q�C��H                                    By
d�&  T          @�=q?�=q�[�?�ffA�G�C�{?�=q�-p�@0��B �\C�                                    By
d��  T          @���@	���dz�?^�RA;�
C�� @	���H��?�z�AծC�/\                                    By
d�r  �          @��@���g
==�G�?�  C���@���Z=q?��HA���C�l�                                    By
e  
�          @��?޸R�z�H=�?���C���?޸R�l��?��A�  C��{                                    By
e�  T          @���?�Q��`��?�{AЏ\C�(�?�Q��1�@5B)  C��R                                    By
e"d  "          @�p�@(���'�@B ��C���@(�ÿ��
@A�B.Q�C��
                                    By
e1
  �          @��@\)�Mp�@A�  C��=@\)�=q@>{B"ffC��3                                    By
e?�  "          @��@���mp�?�z�A�  C�j=@���Fff@�RBz�C��
                                    By
eNV  T          @�
=@��p��?�p�A�C�y�@��L��@z�A��\C��                                     By
e\�  �          @�  ?�{�|��?���A���C��?�{�U@!G�B��C���                                    By
ek�  "          @�  @\)�i��?���A�=qC���@\)�C�
@=qB�RC�{                                    By
ezH  �          @�G�@���k�?���A�ffC�� @���E@=qB �HC��                                    By
e��  �          @�\)@ff�hQ�?���Ai��C�|)@ff�Fff@(�A�z�C��3                                    By
e��  
�          @j�H@��8Q�?&ffA$  C��=@��"�\?�G�A�(�C�L�                                    By
e�:  
�          @p��?�=q�2�\?���AɮC���?�=q�
�H@
=B��C��                                    By
e��  "          @o\)?�\)�4z�?��A�{C���?�\)���@�B�C�*=                                    By
eÆ  
�          @�?���3�
@9��B.�HC�?�녿��
@hQ�Bo�C�N                                    By
e�,  
�          @�ff@(��R�\?˅A�
=C�J=@(��(��@!G�B(�C�aH                                    By
e��  T          @��R@
�H�>�R?��HA�=qC��@
�H���@�
B=qC�&f                                    By
e�x  T          @�p�?\�:�H� ���  C���?\�c�
��p���p�C���                                    By
e�  "          @�{?�z��
=�h���\G�C�� ?�z��H���3�
��HC��                                    By
f�  "          @��<��W��A��'�C�K�<����Ϳ���
=C�=q                                    By
fj  �          @��H>�(��AG��Z�H�A\)C��>�(��{��z���\)C��                                    By
f*  �          @���?B�\�1G��j=q�O�\C���?B�\�qG��'��	�C���                                    By
f8�  T          @��
?xQ��(��p���\ffC���?xQ��^�R�4z����C��H                                    By
fG\  �          @�ff?�33�z��aG��QC���?�33�S33�(Q��
=C��                                     By
fV  �          @�ff?����'
=�U��B�RC�k�?����`���ff� 33C�L�                                    By
fd�  �          @��
?���3�
�HQ��8=qC��H?���i���ff���C�Y�                                    By
fsN  �          @��?�
=�
=�e�Z�C�R?�
=�G��0����C�Q�                                    By
f��  "          @�ff?��R�������\�n��C�j=?��R�A��S33�1=qC��                                    By
f��  �          @���?����ٙ��z=q�qG�C�5�?����5��L(��4G�C��                                    By
f�@  
�          @�
=?���
=�vff�Z��C�Ф?���\���:�H�
=C���                                    By
f��  �          @�  ?����<(��aG��?C��?����x������=qC���                                    By
f��  �          @�?�\)�Dz��R�\�4��C��?�\)�|(��
=q����C��                                    By
f�2  T          @��H?�׿�\)�O\)�K{C��
?���2�\�   �z�C��q                                    By
f��  �          @�Q�@(Q��33�W
=�4�
C��@(Q��@  �#33�Q�C���                                    By
f�~  "          @��R@(���(Q��G���=qC��@(���>{�(��33C���                                    By
f�$  
Z          @|(�@7��,(����
��{C�n@7��%�?E�A4z�C��                                    By
g�  �          @s33@<(���\?
=A  C��@<(����R?��
A�z�C��                                    By
gp  T          @j�H@>{�>u@n�RC�ff@>{��z�?aG�A`  C��R                                    By
g#  �          @n�R@@���
�H>��
@�(�C�"�@@�׿��H?z�HAtQ�C�}q                                    By
g1�  	�          @y��@H���
=q?z�A33C��)@H�ÿ�\)?�p�A��C���                                    By
g@b  �          @g�@>�R���R�W
=�Z=qC�!H@>�R����>�
=@�Q�C�Z�                                    By
gO  
�          @e@9����(��(���*�RC��@9�����#�
���C�9�                                    By
g]�  �          @p  @>{��p����\��C�,�@>{��;�33��z�C���                                    By
glT  �          @j�H@1G��  =�G�?�\C�}q@1G��ff?O\)AR=qC�p�                                    By
gz�  
�          @l��@5���
?\)A��C�]q@5��G�?�G�A��C�@                                     By
g��  �          @qG�@3�
���?E�A<(�C��\@3�
��?�p�A�33C��                                    By
g�F  �          @qG�@3�
�\)?Y��AU�C�� @3�
����?\A�C�L�                                    By
g��  "          @y��@{�G�?�A�(�C��@{��ff@"�\B$�C��                                    By
g��  
�          @y��@Tz��33�
=q�G�C��@Tz��(�=��
?��RC���                                    By
g�8  �          @��@i���ٙ����
����C���@i���ff�k��D��C��                                    By
g��  �          @�z�@fff��녿�����HC�@fff�	����G���  C��
                                    By
g�  �          @�G�@i����G���\)��z�C�z�@i�����
��\)���C��R                                    By
g�*  
�          @�
=@n�R���\��R��\)C�aH@n�R�������G�C���                                    By
g��  "          @���@a녿\(��8Q��\)C�1�@a녿�(��p�� {C�                                    By
hv  "          @�p�@z=q���
���H���C��R@z=q��R����T(�C�q                                    By
h  �          @���@\)�
�H�u�?33C��
@\)�
=�aG��1�C���                                    By
h*�  T          @�\)@h���������p�C��@h���	��>\)?�33C��)                                    By
h9h  �          @��\@vff��  �Y���6�\C��f@vff��
=����Y��C��\                                    By
hH  �          @��H@b�\���>��@�=qC���@b�\�G�?�{As
=C�&f                                    By
hV�  "          @}p�@=p���
=� ���=qC�%@=p���  ������\C��=                                    By
heZ  �          @���@�R���h���\ffC��@�R����R�\�?33C�@                                     By
ht   
�          @���@P�׿޸R����z�C��q@P���{��z���p�C��                                    By
h��  y          @�(�@c�
�!G���{��ffC�XR@c�
�   >�@ȣ�C�w
                                    By
h�L  
�          @��@e�� �׾���
=C�z�@e��!�>�Q�@�\)C�e                                    By
h��  �          @�(�@r�\�
=q>B�\@��C�(�@r�\���R?\(�A6=qC�&f                                    By
h��  
Z          @�(�@p  �\)>��@�z�C���@p  �   ?���Ah��C���                                    By
h�>  �          @�ff@W
=��ÿ\)��Q�C�C�@W
=���>k�@HQ�C���                                    By
h��  �          @�\)@\�����:�H�\)C��R@\���(�=L��?#�
C�W
                                    By
hڊ  �          @���@c33�"�\�&ff�Q�C�7
@c33�'
=>B�\@�HC��\                                    By
h�0  �          @���@Tz��{��  �X��C��f@Tz��*=q�#�
���C���                                    By
h��  �          @�=q@0  �{��
=����C�
=@0  �8Q�L���6=qC��                                     By
i|  �          @�=q@8Q���ÿ��H��{C�"�@8Q��.�R��R�=qC�AH                                    By
i"  �          @�=q@hQ��p��5���C�U�@hQ��z�<�>�{C��3                                    By
i#�  �          @�  @�(���33��Q쿌��C��@�(��˅>�G�@��C�t{                                    By
i2n  �          @��@��H���ͽ����C�T{@��H�\)��Q쿏\)C��                                    By
iA  �          @�@�=q�k�?ǮA�ffC�z�@�=q>�\)?�ffA��@l(�                                    By
iO�  �          @�\)@�Q쿥�@p�A�p�C���@�Q��@ ��A�Q�C��                                    By
i^`  �          @�\)@�G���
=@{A�G�C�K�@�G��O\)@7�Bp�C���                                    By
im  �          @��@�G����@>{B G�C�}q@�G���p�@P��Bp�C���                                    By
i{�  �          @�{@�  ���@Tz�BC�/\@�  <�@`  B�>��
                                    By
i�R  �          @��H@��ͿJ=q@U�B�C��=@���>��R@Z=qB�@��\                                    By
i��  �          @��H@��
��\)@R�\BQ�C���@��
�#�
@^{B�HC��=                                    By
i��  �          @���@�z῕@I��B
�HC��=@�z��G�@W
=B�\C�O\                                    By
i�D  �          @��@�33���
@K�B�HC�N@�33<�@VffBff>�G�                                    By
i��  T          @���@��\����@C33B	p�C��@��\�#�
@N�RB�C���                                    By
iӐ  �          @��\@��׿�
=@=p�B�C�E@��׾.{@K�B��C��{                                    By
i�6  �          @��@�=q���\@?\)B�C�Y�@�=q���
@J=qB�\C��                                    By
i��  �          @�ff@}p����R@;�B
��C���@}p���@P  B�C��3                                    By
i��  �          @���@~{���@0��B	��C��R@~{>��R@3�
B��@�                                      By
j(  �          @���@qG���
=@8Q�BC�f@qG��+�@QG�B!�
C��R                                    By
j�  �          @�z�@�{��z�@J�HB	(�C�/\@�{���
@]p�Bp�C��=                                    By
j+t  �          @�\)@�G����
@P  B
33C�(�@�G��.{@_\)B�C��=                                    By
j:  �          @��@��ÿ���@EB�\C���@��þ��
@W�BG�C��)                                    By
jH�  �          @�z�@�\)�s33@2�\B=qC���@�\)�#�
@<��B�C���                                    By
jWf  �          @���@�ff��Q�@:�HBQ�C�q@�ff�B�\@I��B\)C��3                                    By
jf  �          @�ff@�33�
=@UBp�C�b�@�33�p��@u�B)\)C���                                    By
jt�  T          @��@mp���
@Z�HB\)C��@mp�����@~�RB7p�C���                                    By
j�X  �          @���@Z=q�&ff@\(�B  C�T{@Z=q��\)@�33B@\)C��                                    By
j��  �          @��H@dz����@I��B��C�R@dzῡG�@p  B2�
C�E                                    By
j��  �          @��@g
=��@<(�Bz�C��@g
=����@]p�B*�C���                                    By
j�J  �          @���@]p��p�@G
=B��C��q@]p�����@i��B4�C�0�                                    By
j��  �          @���@fff�
=q@L(�B�C��H@fff���
@mp�B3{C��                                    By
j̖  �          @�  @tz���@UB�C�� @tz�p��@uB1�C��                                    By
j�<  �          @��@�{��@Tz�B��C��@�{�h��@tz�B&C��
                                    By
j��  �          @�ff@��H�޸R@G
=B��C�|)@��H�(��@`  B ��C�l�                                    By
j��  �          @�p�@�Q�Ǯ@R�\B�C�\)@�Q��(�@g�B'��C��\                                    By
k.  �          @�33@�G��Q�@aG�B"
=C�J=@�G�>�33@eB&Q�@�{                                    By
k�  
�          @�ff@{�>�@p  B-�
@�@{�?�33@Y��Bp�A�{                                    By
k$z  �          @��@��
����@w�B,�C�L�@��
?��@l��B#��Aw\)                                    By
k3   �          @�
=@��
���@�G�B*\)C��\@��
?W
=@~�RB'33A.�\                                    By
kA�  �          @�p�@�33��Q�@~�RB)�\C���@�33?n{@x��B$�AAp�                                    By
kPl  �          @��@z=q���@\)B0�C���@z=q=L��@�{B<
=?E�                                    By
k_  �          @���@n�R�"�\@fffBG�C��R@n�R��G�@�
=B<
=C��H                                    By
km�  �          @�=q@L���K�@QG�BC��
@L�Ϳ�(�@��B>Q�C�,�                                    By
k|^  �          @�{@^�R�@��@o\)B(�C��\@^�R��@�  BE�C�4{                                    By
k�  �          @��@e��$z�@u�B#��C�'�@e���(�@�ffBF�\C���                                    By
k��  �          @��
@�Q쿱�@n{B%  C�k�@�Q�\)@~{B2�
C��                                    By
k�P  �          @�ff@g
=��\)@���B<=qC���@g
=<�@��
BI�>�
=                                    By
k��  �          @�\)@j�H��@�
=B>��C�'�@j�H>u@�(�BG�@j�H                                    By
kŜ  T          @�@p  ��{@���B5{C��@p      @�  BB33C��R                                    By
k�B  �          @�G�@�녿�
=@w�B'�
C�E@�녾�@�(�B5�
C��                                    By
k��  �          @��H@\)��@�  B-��C�*=@\)���
@�  B;G�C�q�                                    By
k�  �          @��
@|(���Q�@n{B&ffC���@|(��8Q�@\)B5Q�C���                                    By
l 4  �          @�\)@|(���p�@i��Bz�C��=@|(��8Q�@��HB6ffC�ٚ                                    By
l�  �          @��@�ff�p�@Dz�A��C��=@�ff��\)@g�B  C��                                    By
l�  �          @��\@�\)���H@R�\B	�C���@�\)�z�@j�HBQ�C�Q�                                    By
l,&  �          @��R@�녿�\)@j=qB��C�(�@�녾�p�@\)B1Q�C�h�                                    By
l:�  �          @�ff@�z��p�@_\)B��C��@�z��@w
=B+33C�U�                                    By
lIr  �          @���@�(���{@e�BG�C��q@�(��!G�@\)B.��C���                                    By
lX  �          @�(�@���У�@n{BQ�C��@����Q�@���B.(�C��{                                    By
lf�  �          @�p�@y����@hQ�B33C��f@y���+�@���B6z�C�                                      By
lud  �          @��H@tz��{@Y��Bp�C��@tzῂ�\@{�B3G�C��                                     By
l�
  
�          @�Q�@`  ��H@`  B��C���@`  ��Q�@��HB?��C���                                    By
l��  �          @��@e��4z�@^�RBz�C�� @e�����@�ffB<{C�"�                                    By
l�V  �          @��
@hQ��7
=@hQ�B�C���@hQ����@�33B>��C�Q�                                    By
l��  �          @��@Tz��\@�BF�RC��R@Tzᾊ=q@�Q�B\(�C���                                    By
l��  �          @��@Y�����@�BB��C�~�@Y���Ǯ@���BZ{C���                                    By
l�H  �          @��R@dz����@���B>��C�|)@dzᾨ��@�  BS�C�T{                                    By
l��  �          @�\)@i�����H@�  BCffC�q@i��=#�
@�
=BQQ�?#�
                                    By
l�  �          @�ff@u��\)@�
=B5z�C��\@u�L��@�Q�BFffC�~�                                    By
l�:  �          @��R@|�Ϳ�(�@�\)B4�\C���@|�ͽu@�\)BBz�C���                                    By
m�  �          @�\)@����@|��B%��C���@������@�{B2�C�Z�                                    By
m�  �          @���@�G����\@���B'�RC��)@�G�=u@��RB1��?Q�                                    By
m%,  �          @�
=@������@{�B$��C�q@��>k�@�=qB+�H@?\)                                    By
m3�  �          @�\)@�G���Q�@g
=B��C�5�@�G��aG�@xQ�B"{C��
                                    By
mBx  
�          @��R@�p���
=@Y��BQ�C�y�@�p���z�@k�B��C�>�                                    By
mQ  �          @��@�ff��(�@J=qB�HC�L�@�ff����@^{BG�C���                                    By
m_�  �          @�33@����\)@<��A�ffC�R@���\@N�RBG�C��=                                    By
mnj  �          @��@z=q��  @�B;=qC�33@z=q?�=q@�G�B3�Aw33                                    By
m}  �          @�p�@U��@�G�BU��C��@U?z�H@��RBP=qA�=q                                    By
m��  �          @�@k��\@���BDz�C�@k�?z�H@�B>�Ao�                                    By
m�\  �          @�@^{�333@�p�BMG�C�Ff@^{?8Q�@�p�BM{A<z�                                    By
m�  �          @��@)���&ff@���Bk��C�#�@)��?O\)@�  Bip�A��                                    By
m��  �          @�@5����@���BTC�  @5���p�@��BoffC�O\                                    By
m�N  �          @�ff@7���ff@�\)Bd�C��{@7�>��H@�=qBk��A
=                                    By
m��  �          @�=q@=p���ff@�  Blz�C���@=p�?�33@�z�Bd�A�p�                                    By
m�  �          @��@J=q��Q�@�33Bb��C���@J=q?�
=@�
=BY��A�\)                                    By
m�@  �          @���@7��k�@���Bp�RC���@7�?�{@��HBc
=A�Q�                                    By
n �  
�          @���@,(����
@�=qBwC���@,(�?��@�p�Bk(�A���                                    By
n�  �          @��H@4z�˅@�G�B_�HC�Ff@4z�#�
@���Bs\)C���                                    By
n2  �          @��R@2�\>L��@�p�Bq��@��
@2�\?�p�@��BZ=qA��\                                    By
n,�  �          @�@7
=?n{@���Bg(�A��@7
=@Q�@�  BC(�B�R                                    By
n;~  �          @�@3�
?��@�B`��AҸR@3�
@1�@�Q�B5��B3{                                    By
nJ$  �          @�@��>Ǯ@��RB5  @���@��?�
=@xQ�B#z�A�{                                    By
nX�  �          @���@��\?�@�\)B7\)@�
=@��\?�@w
=B#p�A���                                    By
ngp  �          @��\@�Q�>�
=@��RB8�@��R@�Q�?��H@w�B&Q�A��R                                    By
nv  �          @�33@�p�<�@u�B#��>�p�@�p�?�(�@hQ�Bp�Av�H                                    By
n��  �          @���@��;��@n{B ��C�N@���?h��@g�B=qA:�H                                    By
n�b  �          @\@�G���\@l��B33C��@�G�?(��@k�B{@�(�                                    By
n�  �          @��@�{���
@s�
B=qC�3@�{?c�
@n{B�
A+33                                    By
n��  T          @�=q@�  �Ǯ@p��B
=C��@�  ?L��@l(�B��A(�                                    By
n�T  �          @��@�p��z�@l(�B=qC�z�@�p�?��@k�B{@�Q�                                    By
n��  �          @�(�@�p���\)@w�B$C���@�p�?��@l��B{Af�R                                    By
nܠ  �          @���@��
?�\@�Q�B0�@��@��
?�(�@i��BG�A��
                                    By
n�F  �          @�  @�
=>��@uB(�R@�z�@�
=?��@`��B33A�                                      By
n��  �          @�@fff?s33@g
=B0ffAm@fff@G�@H��B�
A��H                                    By
o�  �          @�z�@N�R?�ff@qG�B=�A��@N�R@��@K�BQ�B�                                    By
o8  �          @�p�@Z=q?aG�@s33B<�\Ag\)@Z=q@ ��@UB �\A�                                      By
o%�  �          @�p�@i��>��@~�RB=��@��\@i��?�p�@mp�B-
=A�=q                                    By
o4�  �          @��@��H>�{@��B4=q@�  @��H?˅@s33B#A�(�                                    By
oC*  �          @�p�@�Q쾙��@g�B'�HC��q@�Q�?Tz�@b�\B#(�A;33                                    By
oQ�  �          @��@s�
=�\)@[�B'��?�ff@s�
?�\)@O\)B�HA�\)                                    By
o`v  T          @�ff@k�>��@dz�B/�@��H@k�?��R@QG�B
=A�=q                                    By
oo  �          @���@i��?J=q@l��B2�HABff@i��?��@Q�BQ�A�z�                                    By
o}�  �          @��R@fff?&ff@l��B5�\A#\)@fff?�  @U�B��A�{                                    By
o�h  �          @�@hQ�?��@j=qB3�A	�@hQ�?��@S�
B�A���                                    By
o�  �          @�ff@W�?@  @c33B7(�AHQ�@W�?�ff@I��B��A��H                                    By
o��  �          @�@A�?���@mp�BD{A���@A�@�@L��B"G�B�H                                    By
o�Z  �          @��@;�?��@l(�BC�RA�(�@;�@��@G
=Bz�Bp�                                    By
o�   T          @��R@h��>�@l(�B4�@�{@h��?�=q@W
=B!33A�                                    By
oզ  �          @�33@�녽#�
@^{B!�C��R@��?��\@S�
BffAa                                    By
o�L  T          @�G�@�녽���@XQ�B�HC�Ff@��?n{@P  B�AN�H                                    By
o��  T          @��@�33<��
@P  Bz�>��
@�33?�G�@E�B�\A]                                    By
p�  �          @��\@�Q�=�@`��B$��?ٙ�@�Q�?�
=@S�
BffA��                                    By
p>  T          @��@{�>�  @g�B*33@fff@{�?��@W�B=qA�(�                                    By
p�  T          @��
@p��?�=q@i��B,{A�ff@p��@��@I��B33A�33                                    By
p-�  �          @��
@��?5@s33B*�HA�R@��?�=q@Z=qB�RA�Q�                                    By
p<0  �          @���@��\?^�R@�Q�B/p�A@��@��\@�\@c33B�A��                                    By
pJ�  �          @���@~�R?�{@�
=B2
=A�Q�@~�R@3�
@a�BB{                                    By
pY|  "          @�@`��?�p�@�BBQ�A��
@`��@?\)@l��B{B!                                    By
ph"  �          @��@\��?��@�  BI�
A�
=@\��@+�@w�B&33Bz�                                    By
pv�  �          @���@X��?���@��BK�RA�=q@X��@'�@xQ�B(�\Bz�                                    By
p�n  �          @��@b�\?��\@�(�BG�RA���@b�\@33@w�B*
=B�
                                    By
p�  �          @�Q�@j�H?Y��@�33BDz�AP��@j�H@�@x��B*=qA�z�                                    By
p��  �          @�\)@�����
@~�RB0=qC��H@��?�
=@s33B&p�A�=q                                    By
p�`  �          @��@~{�B�\@z�HB0�\C��
@~{>�@~{B333@��                                    By
p�  �          @��@z=q����@o\)B%�
C��q@z=q���@�G�B7(�C�
=                                    By
pά  �          @�p�@�녿��@l(�B"�C��{@�녾aG�@|(�B033C�|)                                    By
p�R  
�          @�
=@�ff��(�@Q�B�C�s3@�ff�fff@n{B#p�C��)                                    By
p��  �          @��@xQ��i��@=qA\C�c�@xQ��0��@X��BC�K�                                    By
p��  �          @�
=@�Q��mp�@��A�  C��)@�Q��4z�@Y��B
=C�o\                                    By
q	D  �          @�33@�{�b�\@#�
A�=qC��@�{�'�@`��BffC��                                    By
q�  �          @�@�p��I��@>{A�\C�t{@�p���@qG�Bp�C��H                                    By
q&�  �          @�G�@�
=����@�G�B�C���@�
=��@�z�B1G�C���                                    By
q56  �          @�Q�@�ff���@�B'�\C�L�@�ff��Q�@���B2�C�q�                                    By
qC�  �          @�{@�=q�/\)@,��A�G�C�/\@�=q����@X��BQ�C���                                    By
qR�  G          @ƸR@�
=�33@P��A��\C�  @�
=��p�@s33B�C���                                    By
qa(  '          @�  @��ÿ�@��BEffC�  @���?\(�@�{BB�HAA��                                    By
qo�  �          @�\)@��ÿ�p�@��B=�C�~�@���>.{@�
=BF  @�R                                    By
q~t  �          @�\)@�  ���@�B5ffC��@�  >��R@�G�B;G�@�
=                                    By
q�  �          @�
=@u�fff@�z�BR
=C�b�@u?(��@�p�BT33A33                                    By
q��  �          @�  @tz�=L��@�p�BP?O\)@tz�?�  @�{BC(�A��                                    By
q�f  �          @�Q�@_\)�B�\@�(�B\ffC�� @_\)?J=q@��
B\(�AJ�R                                    By
q�  �          @�  @q녿Q�@�(�BNz�C��=@q�?(��@��BO�HA                                    By
qǲ  �          @�
=@R�\�u@�Bb  C��{@R�\?��@�\)Be�A%p�                                    By
q�X  �          @�ff@b�\���@�
=BU�\C���@b�\>�@��B[
=@�                                    By
q��  �          @Å@QG�����@���B^�C���@QG�>�ff@�(�Be
=@��H                                    By
q�  �          @��@]p���  @��BX�RC��
@]p�?�\@��B]z�Ap�                                    By
rJ  �          @�(�@L�Ϳ�
=@�=qB`33C���@L��>�Q�@�{Bh�@��H                                    By
r�  �          @�\)@>�R��\@�BsffC�%@>�R?���@��HBl��A�G�                                    By
r�  �          @�p�@AG��^�R@��Bm\)C���@AG�?5@��\Bo�AS�
                                    By
r.<  �          @�
=@a녾���@��B]{C��=@a�?���@�\)BT�RA�                                    By
r<�  �          @ƸR@`  ���H@��B]\)C��@`  ?��\@���BX\)A��                                    By
rK�  
�          @�(�@o\)���@��HBP(�C��
@o\)?aG�@�G�BM(�ATQ�                                    By
rZ.  �          @Å@l(��Tz�@��\BO�
C��@l(�?(�@��BQ�
A                                    By
rh�  �          @�{@E�>�{@���Bn�@�
=@E�?�@�{BX
=A�\)                                    By
rwz  �          @�G�@C�
��R@�33Bj
=C�<)@C�
?aG�@��BgQ�A\)                                    By
r�   �          @���@E����@��
Bkz�C�3@E?���@�B^�A���                                    By
r��  �          @�\)@/\)���R@�G�By��C�Ф@/\)?�p�@��Bo  A�33                                    By
r�l  �          @�z�@p�?333@���B�=qA
=@p�@(�@��B_=qB&�                                    By
r�  �          @�z�@<�ͼ�@��\Boz�C���@<��?�
=@�(�B`ffAθR                                    By
r��  �          @��@6ff��Q�@��
Bs�C��@6ff?���@�Be�AΣ�                                    By
r�^  "          @�(�@#33>��@�Q�B~�A&�H@#33?���@���Bc  B��                                    By
r�  �          @�(�@��>�33@�33B�L�A��@��?�\)@���Bkp�B��                                    By
r�  
�          @�p�@{>���@��B��A��@{?�@���BhQ�B(�                                    By
r�P  T          @�p�@
�H?z�@�
=B�p�Anff@
�H@
=@��\Bl�B1{                                    By
s	�  �          @�z�?��?&ff@�=qB�8RA��R?��@p�@���BtffBL{                                    By
s�  �          @�?���>8Q�@�33B�\)@�Q�?���?�G�@�=qB~�B((�                                    By
s'B  �          @�ff@�>8Q�@�G�B��)@��R@�?޸R@�Q�Bx(�Bp�                                    By
s5�  �          @���@�=q?O\)?L��A��A!@�=q?�G�?
=q@��AG
=                                    By
sD�  �          @���@��@��(���33A�=q@��?�zῙ���@z�A��\                                    By
sS4  �          @��@��R@3�
?333@�\)A�  @��R@8�ý���(�A�                                    By
sa�  �          @�33@Q�?(�@�=qB�� Af{@Q�@�@�Bep�B$��                                    By
sp�  �          @��\@��>�@��HB�#�A?
=@��?��H@��Bl�B#�\                                    By
s&  �          @�33@<(�?�ff@�ffB[
=A���@<(�@3�
@��\B4ffB.�                                    By
s��  �          @�@��?(��@��\B�Ao\)@��@�@�Bb��B#ff                                    By
s�r  �          @��?�Q�?��@�33B��A�p�?�Q�@��@��RBy�\BN                                    By
s�  T          @��@n�R?J=q@�{BE\)A?
=@n�R@   @�G�B.A�G�                                    By
s��  �          @���@Fff>Ǯ@��Bf�@�{@Fff?޸R@�33BQQ�A��H                                    By
s�d  �          @���?
=q>��H@�p�B�G�B(G�?
=q@33@��B�z�B�B�                                    By
s�
  �          @��H��p�?333@�\)B�=qB�q��p�@��@��B�=qBƙ�                                    By
s�  �          @�zῂ�\���H@�B��RCrxR���\�\@�Q�B��{CHJ=                                    By
s�V  �          @�p��W
=���@�(�B��\C����W
=�#�
@��B��CY��                                    By
t�  �          @\?�녿0��@��B�  C��f?��?s33@���B��3A��
                                    By
t�  �          @���@   ��ff@��B���C�� @   ?���@�
=B}A�G�                                    By
t H  �          @��@C�
�u@�=qBp�C��
@C�
?�p�@�{Bf(�A�33                                    By
t.�  �          @��@s�
��Q�@�(�BJ33C�S3@s�
?�Q�@�\)BA(�A�=q                                    By
t=�  �          @�33@j�H    @��HBSp�C���@j�H?�=q@��BH=qA��H                                    By
tL:  �          @��@g
=���@��BL(�C��q@g
=>k�@���BS��@hQ�                                    By
tZ�  �          @�Q�@<�Ϳ�{@�p�BbffC���@<��=#�
@�33Bp  ?=p�                                    By
ti�  
(          @��@�=q����@\)B$��C���@�=q���R@�\)B1G�C��                                    By
tx,  �          @��
@����ff?��AS�C�.@�����z�?�{A�(�C�O\                                    By
t��  �          @�(�@�ff��@  A�G�C�^�@�ff�@  @#33A��C�޸                                    By
t�x  �          @�p�@���h��@z�A�G�C�/\@����  @�RA���C���                                    By
t�  �          @�p�@�(���=q?�z�A]��C��R@�(��&ff?�33A��C���                                    By
t��  �          @���@��
����?�p�AA�C��
@��
�Q�?�G�An�\C��{                                    By
t�j  �          @�{@�=q�333?c�
A	C��\@�=q��G�?��A"=qC�ٚ                                    By
t�  �          @�@��\�O\)?
=q@�G�C��@��\�!G�?=p�@�RC��f                                    By
t޶  �          @�G�@�p��c�
?��@�{C���@�p��333?O\)A�C�|)                                    By
t�\  �          @�z�@�=q�G�?��A��C���@�=q���
?��A�  C�^�                                    By
t�  �          @��@�(���������}p�C�q�@�(����R=�?�Q�C�C�                                    By
u
�  �          @��@�(���p���Q�^�RC�S3@�(���
=>�G�@�\)C��=                                    By
uN  T          @�
=@�z��&ff?L��A=qC��f@�z��33?��HA��C�,�                                    By
u'�  �          @�ff@fff��\)?��HA��C�aH@fff�\)@(�A��C���                                    By
u6�  �          @�@^�R�   �Vff�
=C�*=@^�R�S33�$z���\)C�G�                                    By
uE@  �          @�
=@K��+��y���,C���@K��g��C�
���C���                                    By
uS�  �          @��@G���R�n{�,�C���@G��W��;��=qC�^�                                    By
ub�  �          @��@^{�p��^{��HC�W
@^{�R�\�,�����C�G�                                    By
uq2  �          @�33@c�
�!G��x���&�HC�]q@c�
�\���E���C��3                                    By
u�  �          @��@j�H�AG��S33�{C�H�@j�H�q������p�C�
                                    By
u�~  �          @�p�@����Z=q����z�C��)@����tzῈ���.ffC�<)                                    By
u�$  �          @��
@{��u�xQ��   C�Ф@{��}p�=�\)?0��C�`                                     By
u��  �          @�(�@xQ��~�R�   ��{C�  @xQ��~�R?��@���C�%                                    By
u�p  �          @��@fff���\>\@uC��)@fff���\?��RAt��C���                                    By
u�  �          @���@z�H���=���?��
C�b�@z�H���?���A?�C���                                    By
u׼  T          @��R@|(����H�����ffC���@|(���=q?(�@�33C�                                    By
u�b  �          @��
@�(��Y���aG����C�
@�(��`��<��
>uC���                                    By
u�  �          @�ff@��H�0  �����6�HC�8R@��H�<(��\�vffC�Y�                                    By
v�  �          @���@�(��C�
��Q�c�
C��3@�(��>�R?333@�
=C�J=                                    By
vT  �          @��H@�p��aG�?G�@��\C��{@�p��Mp�?�33A��HC�H                                    By
v �  �          @�=q@X����G�?�G�A"ffC�XR@X����(�@��A���C���                                    By
v/�  �          @�(�@w���  ?�{A/\)C�+�@w��tz�@
=qA�33C���                                    By
v>F  �          @��@�
=�P��?��A3\)C��\@�
=�7
=?�
=A�{C�aH                                    By
vL�  �          @��@��N{@�A��
C�@��#�
@;�A�(�C��                                    By
v[�  �          @�
=@�Q��w�@�A���C��@�Q��Mp�@A�A�ffC���                                    By
vj8  �          @�
=@�G��z=q?�{AR�RC��\@�G��\(�@33A��C���                                    By
vx�  �          @��@����l(�?�p�A;�C�Ǯ@����P��@�A��C�y�                                    By
v��  �          @�\)@�z��g
=?��A$(�C�{@�z��N{?�Q�A�G�C��q                                    By
v�*  �          @�Q�@�(��HQ�?�A\  C���@�(��+�@
=qA��C���                                    By
v��  �          @�=q@�{�8Q�?�(�A��C��)@�{�33@(Q�A��C���                                    By
v�v  �          @Å@�ff�'
=@z�A��C�{@�ff����@8��A�z�C�:�                                    By
v�  �          @��@���9��@�RA��C�xR@���  @8Q�A�  C�w
                                    By
v��  �          @�G�@����
�H@333A܏\C��@��Ϳ�z�@P  B
=C���                                    By
v�h  �          @�  @��
�=q@!�A�  C��
@��
���H@C33A��RC�W
                                    By
v�  �          @�33@�G��'
=@,(�A�(�C��@�G���\)@P  B\)C�W
                                    By
v��  �          @�@�p��9��@�A��C��H@�p��\)@>{A�Q�C��=                                    By
wZ  �          @�G�@���.{@#33A�(�C��@��� ��@H��A�
=C�~�                                    By
w   �          @�p�@%?�@�{B`
=B  @%@:=q@�33B:  BA33                                    By
w(�  �          @�G�@�
@\)@�33Ba��B=p�@�
@S33@z�HB4Bg�
                                    By
w7L  �          @�
=?�{@;�@��\BN�B��?�{@u@P  BffB��H                                    By
wE�  
�          @�  ?+�@C33@xQ�BL��B�B�?+�@y��@AG�Bp�B��=                                    By
wT�  �          @�(�?Q�@\��@x��B>p�B�8R?Q�@�G�@<(�B�\B�Q�                                    By
wc>  �          @��
@  ?�Q�@�=qBs\)A�ff@  @��@�z�BQ��B4�                                    By
wq�  �          @��?�G�?��H@�z�B|��BQ�H?�G�@E�@���BL�B                                    By
w��  �          @�?
=q@i��@���BE��B�{?
=q@�=q@S33Bz�B�z�                                    By
w�0  �          @�z�<�@��
@�B5p�B��<�@�Q�@Dz�A��B�B�                                    By
w��  �          @�  �u@{�@�p�B:ffB�{�u@�=q@G
=B33B�Ǯ                                    By
w�|  �          @���=L��@tz�@�G�BAQ�B���=L��@�\)@P��B
33B��f                                    By
w�"  �          @�z�>�z�@U@��RBM�
B�\>�z�@��@R�\B
=B�.                                    By
w��  �          @�  =�@Fff@���B^B��{=�@��H@k�B'��B���                                    By
w�n  �          @��H>�G�@5@�=qBq��B�W
>�G�@}p�@�  B;p�B�B�                                    By
w�  �          @�33?��R@&ff@���BkB��)?��R@i��@�G�B9(�B�ff                                    By
w��  �          @�@   @��@��Bb��BG�@   @[�@�G�B6  Bn�                                    By
x`  �          @�(�@��?��
@�Q�Bl�RB��@��@:=q@�ffBG  BI��                                    By
x  �          @�33?�(�@3�
@�=qB^z�Bjff?�(�@w
=@���B.z�B�
=                                    By
x!�  �          @���@
=@\)@��\Bf
=B:z�@
=@S33@�B;\)Bez�                                    By
x0R  �          @�G�@�@��@�z�BW33B3(�@�@N{@o\)B-��BZ��                                    By
x>�  T          @�p�?���@;�@��BW�BwQ�?���@z=q@p��B'
=B�=q                                    By
xM�  �          @�33?��@7
=@��HB^{B��f?��@u@s�
B,�B���                                    By
x\D  �          @���?fff@J�H@�{BU�B�W
?fff@��@eB!�
B�L�                                    By
xj�  �          @��R?��@j=q@�  B<��B���?��@�\)@C�
BffB��                                    By
xy�  T          @��?�R@;�@��Bb�B��?�R@x��@qG�B.�HB��                                    By
x�6  �          @��>���@���@y��B/33B���>���@��\@7�A�\)B�p�                                    By
x��  �          @�  >���@�33@\)B0�\B��>���@���@=p�A�ffB��q                                    By
x��  �          @�=q?p��@���@uB$��B�Q�?p��@���@1G�A�33B��                                    By
x�(  �          @�녿
=q@��@\��B�
B�Q�
=q@�\)@��A��B�k�                                    By
x��  
�          @��׾k�@��
@UB�B�  �k�@�Q�@G�A�
=B�=q                                    By
x�t  �          @�\)�aG�@��\@>�RB
=B��{�aG�@�z�?��A�ffB���                                    By
x�  �          @�����R@��\@@  B�B��쾞�R@�z�?�33A�(�B���                                    By
x��  �          @�
=��\@��@,(�A�33B�B���\@�
=?ǮA��\B�#�                                    By
x�f  �          @�{��@��@%A�Q�B�p���@�ff?�(�A|  B�\)                                    By
y  �          @�Q쾞�R@���@.�RA���B������R@�(�?У�A��RB��H                                    By
y�  �          @��\?�@���@5A��RB�\)?�@���?�(�A�33B���                                    By
y)X  �          @�33?Q�@��@5�A�B�G�?Q�@�  ?�(�A�G�B�(�                                    By
y7�  �          @�Q�?z�@�33@ ��A�\)B�p�?z�@���?���Ai�B��=                                    By
yF�  T          @���=L��@��@=p�B�RB�=L��@�ff?�\)A�Q�B��f                                    By
yUJ  �          @���?B�\@��@A�B  B�(�?B�\@��?��HA�ffB�(�                                    By
yc�  �          @���?Y��@�z�@>�RB��B�
=?Y��@�{?���A�ffB�k�                                    By
yr�  T          @���?J=q@���@%�A��
B�\)?J=q@�  ?��A�
=B��                                    By
y�<  T          @��?!G�@��@3�
A���B�
=?!G�@��?��
A�B���                                    By
y��  �          @��
?(��@��R@8Q�B33B�8R?(��@�
=?���A��B��                                    By
y��  �          @��?���@e@fffB*33B�?���@���@0  A���B�\                                    By
y�.  �          @��H@\)?��@e�BK  A��@\)@G�@J�HB-
=B)p�                                    By
y��  �          @��@�(�����@J�HB
=C���@�(��+�@X��B�RC��)                                    By
y�z  �          @�
=@�ff���R@XQ�B	C�7
@�ff�=p�@g�B{C���                                    By
y�   �          @�p�@�
=��  @UB	z�C��
@�
=��\@a�Bp�C��                                    By
y��  �          @��@�  ��Q�@?\)A�G�C�7
@�  ���\@QG�B  C��\                                    By
y�l  �          @��@�  ����@333A�z�C��H@�  ��Q�@G�B�C���                                    By
z  �          @�G�@���J�H@��A��C�s3@���&ff@@��A�\)C��3                                    By
z�  �          @�p�@�(��^{@Q�A��C���@�(��<��@3�
A���C��R                                    By
z"^  T          @���@�G��k�?��A�G�C���@�G��L��@'
=A�G�C�g�                                    By
z1  T          @�
=@����u@(�A��RC�\)@����S33@<(�A�{C�e                                    By
z?�  �          @�
=@�
=����@�\A��C��H@�
=�aG�@5�A�\)C�]q                                    By
zNP  �          @�=q@����Q�?���A|(�C��)@���r�\@,(�A�C�`                                     By
z\�  �          @�33@�����\?�{AYC���@���y��@p�A��RC�+�                                    By
zk�  �          @�\)@����G�?�\)A<z�C���@���z�H@{A�(�C��                                    By
zzB  �          @��@��R���;8Q��C��3@��R��33?8Q�@љ�C��H                                    By
z��  �          @��@����{����33C���@����ff>�(�@z�HC��                                     By
z��  �          @Ǯ@\)��
=�J=q��G�C�R@\)��G�>8Q�?��C���                                    By
z�4  �          @�\)@�
=��=q�k��
=C���@�
=����?&ff@�ffC�q                                    By
z��  �          @ȣ�@~�R��
=�u�p�C��@~�R��=q    =L��C�                                    By
zÀ  �          @��@i�����R��\)�H��C�*=@i����(���33�N{C���                                    By
z�&  �          @�  @��\��Q����z�C��f@��\��?Q�@�  C�˅                                    By
z��  T          @Ǯ@��H��\)<�>���C���@��H��z�?^�RA (�C���                                    By
z�r  �          @���@��\��Q�>k�@C�*=@��\�xQ�?�G�A��C���                                    By
z�  �          @ə�@�  ��(�>#�
?��HC��@�  ��Q�?xQ�A��C���                                    By
{�  �          @ʏ\@�{���=�Q�?O\)C��{@�{��z�?k�Ap�C�K�                                    By
{d  �          @ʏ\@�G��u<�>�z�C�W
@�G��p��?G�@�=qC��H                                    By
{*
  �          @��H@�z��mp�>L��?�C�{@�z��fff?k�Ap�C�}q                                    By
{8�  �          @���@����n{>���@.{C��@����fff?�G�Az�C�G�                                    By
{GV  �          @�Q�@�p��vff>���@?\)C��
@�p��n{?��Az�C�u�                                    By
{U�  �          @�G�@�{�u?
=@�z�C�@�{�j=q?���AA��C��)                                    By
{d�  �          @��@�{�u�?W
=@�C��@�{�fff?ǮAe�C��
                                    By
{sH  �          @ʏ\@�=q�l��?B�\@�z�C���@�=q�_\)?���AT  C��R                                    By
{��  �          @��H@��H�mp�?+�@���C��\@��H�aG�?�{AF{C���                                    By
{��  �          @��
@���l(�?W
=@�C�{@���^{?\A\��C��                                    By
{�:  T          @�{@��H�k�?���AF�\C��@��H�W
=@33A��RC�H�                                    By
{��  �          @��@���o\)?��HA-C��f@���\��?��A���C��                                     By
{��  �          @ȣ�@����xQ�>�p�@[�C��\@����o\)?��A ��C�O\                                    By
{�,  �          @�  @���u>Ǯ@eC��R@���l��?���A!�C�z�                                    By
{��  �          @ʏ\@�  �r�\?&ff@�p�C�l�@�  �g
=?��ADQ�C�)                                    By
{�x  �          @�
=@��R�e?��AEC���@��R�QG�@G�A��C���                                    By
{�  �          @��@���b�\?���Ab�HC��@���L(�@(�A�  C�H                                    By
|�  �          @�@�=q�e�?�33An=qC�g�@�=q�N{@�A�  C�Ф                                    By
|j  �          @�p�@�\)�n�R?��A^�RC��3@�\)�X��@��A�(�C��                                     By
|#  T          @ʏ\@��
�y��?xQ�A{C���@��
�j=q?�z�At��C��f                                    By
|1�  �          @�G�@��s33?aG�AG�C�/\@��e�?�ffAe�C��                                    By
|@\  �          @��
@�ff�_\)?���A-�C��@�ff�N{?�A��
C�                                      By
|O  �          @�  @�
=�;�?��A�C��@�
=�#33@��A��RC��H                                    By
|]�  �          @�=q@�G��H��?���Ab{C�:�@�G��3�
@��A��
C��3                                    By
|lN  �          @У�@��
�R�\?˅Ab{C�>�@��
�=p�@��A�(�C���                                    By
|z�  �          @��@��
�`��?˅Af=qC�Ǯ@��
�J�H@(�A���C�R                                    By
|��  �          @�@�G��fff?�(�Ax(�C�8R@�G��O\)@z�A���C��)                                    By
|�@  �          @�p�@����j=q?�G�AYC�@����U@Q�A���C�G�                                    By
|��  �          @�  @�
=�\(�@Q�A�G�C���@�
=�AG�@,��A�
=C�Z�                                    By
|��  T          @�G�@��R�\��?xQ�A��C��R@��R�N�R?�ffA\(�C��\                                    By
|�2  �          @���@���hQ�?�{Ap�C��3@���XQ�?�(�At��C��H                                    By
|��  �          @���@�33�e?�z�ALQ�C�o\@�33�S33@ ��A�ffC��{                                    By
|�~  �          @�  @���g
=?�Q�A{�
C��\@���QG�@�\A�C���                                    By
|�$  �          @���@����n�R?�\A�{C�T{@����XQ�@Q�A�=qC��R                                    By
|��  �          @�  @���w
=?�A��HC���@���_\)@{A��
C�H�                                    By
}p  
�          @�=q@���|��?�=qA��HC���@���e@{A�Q�C��                                    By
}  �          @�33@��R��ff?�z�Ar�HC�^�@��R�w
=@A�33C���                                    By
}*�  �          @�@�{�x��@�A���C�&f@�{�_\)@,��A�ffC��=                                    By
}9b  �          @��@�ff�u�@A�  C�ff@�ff�[�@,(�A���C��                                    By
}H  �          @��@����r�\?�
=A�(�C��{@����Z�H@!�A��C�@                                     By
}V�  �          @�33@�p��q�@�\A�=qC��f@�p��X��@(��A�ffC�                                    By
}eT  �          @ʏ\@�
=�n�R?���A��C���@�
=�W
=@"�\A�
=C�AH                                    By
}s�  �          @�
=@�(��o\)@�A�{C�C�@�(��W
=@'
=A�(�C���                                    By
}��  �          @���@�G��b�\@��A��C�w
@�G��H��@+�A�ffC�                                    By
}�F  �          @�ff@�\)�\��@(�A�G�C��f@�\)�C33@.{A��C�AH                                    By
}��  `          @�\)@�{�l��?�A�
=C���@�{�W
=@=qA�  C��                                    By
}��  
�          @θR@��R�z�H?��A8��C���@��R�j�H?�33A��C��                                    By
}�8  
�          @��
@�
=�q�?�  A4��C�Y�@�
=�b�\?�=qA�C�E                                    By
}��  
�          @��@��
����?�{A�HC�+�@��
�s�
?�p�Az=qC���                                    By
}ڄ  �          @���@�(���  ?�=qA�C�Y�@�(��q�?ٙ�AuC�#�                                   By
}�*  
�          @θR@���\)?��A:ffC�p�@���o\)?�z�A��
C�Y�                                   By
}��  "          @У�@��
=q@~{Bz�C��R@��\@�Q�B#��C��                                    By
~v  �          @�Q�@����  @��\B(�C��R@����˅@�(�B)  C�]q                                    By
~  "          @�p�@�z��(�@�Q�B�C�U�@�z��G�@���B,{C��R                                    By
~#�  �          @�p�@�  �{@��
B��C�w
@�  �Ǯ@��B%�C��                                    By
~2h  "          @љ�@�ff�!�@��\B
=C�.@�ff���@��B)��C��=                                    By
~A  �          @�  @��
�2�\@z=qB{C��3@��
�
�H@�G�B%Q�C���                                    By
~O�  
�          @��
@�z��2�\@s33B(�C��H@�z���@�p�B�\C�g�                                    By
~^Z  �          @��H@�\)�0  @j�HBp�C��=@�\)�
�H@�G�B\)C��\                                    By
~m   "          @У�@�
=�/\)@c�
BQ�C��@�
=��@{�B(�C���                                    By
~{�  T          @�
=@�ff� ��@g�B�RC��R@�ff��Q�@|��BQ�C���                                    By
~�L  T          @�\)@��
�0��@g
=B\)C���@��
�(�@~�RBffC�O\                                    By
~��  �          @���@�p��E@g�B�C�@ @�p��!�@�G�B��C�˅                                    By
~��  T          @�ff@�33�U@I��A��C���@�33�5@fffB��C�Ф                                    By
~�>  T          @�{@����W�@O\)A�=qC�aH@����7
=@l(�BffC���                                    By
~��  
(          @�33@�ff�Y��@HQ�A�z�C�\@�ff�:=q@eB��C��                                    By
~ӊ  
�          @�=q@����I��@AG�A�p�C��@����+�@\��A�33C���                                   By
~�0  �          @љ�@����U�@2�\AɮC��@����9��@N�RA��HC��{                                   By
~��  "          @ʏ\@��,(�@S�
A�=qC��@��(�@j=qB=qC�p�                                    By
~�|  �          @ƸR@�G��>{@@��A��C�ff@�G��!G�@Y��BC��                                     By
"  T          @˅@�ff�J=q@6ffAծC��@�ff�.�R@P��A�  C��                                    By
�  �          @�@����J�H@E�A�C��q@����-p�@`  B  C��                                    By
+n  T          @Ϯ@����333@l��B(�C�'�@������@���B�RC�                                    By
:  "          @�@�(��0��@q�BffC��@�(��p�@��
B {C��q                                    By
H�  �          @Ϯ@����z�@�Q�B�RC�z�@�����  @�G�B%��C�|)                                    By
W`  T          @�ff@�ff��(�@�{B.�RC�q�@�ff����@��B:=qC��                                    By
f  "          @�p�@�p����@�(�B =qC�q@�p��Ǯ@�(�B,=qC�O\                                    By
t�  
�          @�\)@�p��
=@�z�BG�C��
@�p����
@�p�B+ffC��                                    By
�R  "          @���@��H�{@tz�Bz�C�(�@��H��Q�@��B#z�C���                                    By
��  
�          @Ǯ@�{�	��@s33B=qC��@�{��\)@���B"33C��R                                    By
��  "          @���@��p�@vffB\)C��)@���
=@��B#��C��R                                    By
�D  
�          @�Q�@����#33@g
=B��C��@����33@z=qB��C�]q                                    By
��  �          @�p�@���p�@n{B�RC�|)@�����H@~�RB=qC�R                                    By
̐  T          @��
@���Q�@qG�Bp�C��q@����\)@���B��C�n                                    By
�6  T          @�z�@���   @mp�B�C�}q@���   @�  B{C��f                                    By
��  T          @θR@����8Q�@�RA�  C�b�@����"�\@5�AхC��                                     By
��  �          @��H@����u>#�
?�
=C�N@����r�\?&ff@��
C�}q                                    By
�(  "          @�@�Q��\)>��H@��HC�� @�Q��x��?}p�Ap�C�{                                    By
��  �          @�@�Q��tz�?�G�A5��C�T{@�Q��hQ�?޸RAz�RC��                                    By
�$t  	�          @�33@�ff�z=q?&ff@��
C��)@�ff�s33?���A$��C�C�                                    By
�3  �          @ʏ\@�{�u�?h��A�C�!H@�{�k�?���AJffC���                                    By
�A�  �          @�=q@��s�
?���A�
C�&f@��i��?��
A`z�C���                                    By
�Pf  "          @���@�
=�l��?L��@�(�C��=@�
=�dz�?�  A8��C�#�                                    By
�_  T          @�(�@����Y���0���θRC�H@����]p���=q� ��C�Ǯ                                    By
�m�  �          @�ff@��H�a�>�?�p�C���@��H�_\)?\)@�{C�Ǯ                                    By
�|X  
Z          @�ff@�G��e�>�@���C�O\@�G��_\)?c�
Az�C��                                     By
���  T          @ə�@�p��z�H�k��
=C���@�p��z�H>u@�C��
                                    By
���  T          @˅@���z�H��p��S�
C���@���|(�=���?k�C���                                    By
��J  T          @��
@����녿+���  C��@������8Q�У�C���                                    By
���  �          @ʏ\@�{��
=��{�k
=C�33@�{��(�������
C���                                    By
�Ŗ  �          @ə�@�����{��G��9�C���@�����녿B�\��\)C�*=                                    By
��<  
�          @�G�@�����\)��{�!�C�l�@������\�(���G�C�R                                    By
���  �          @ə�@���z=q�k��	��C���@���z=q>aG�?��RC��{                                    By
��  �          @ʏ\@����s�
?\)@�=qC�c�@����n{?}p�AQ�C��
                                    By
� .  "          @�33@��H�l(�?n{A�HC�@��H�c�
?���AC�C��H                                    By
��  T          @���@���Y��?��AAG�C���@���N�R?��HAx��C�/\                                    By
�z  �          @˅@��Y��?�AO
=C�\)@��N{?�ffA�33C��                                    By
�,   T          @��H@�Q��vff=���?c�
C�9�@�Q��tz�?�@���C�Y�                                    By
�:�  T          @��
@�
=�\)��G��xQ�C��q@�
=�~�R>���@?\)C���                                    By
�Il  �          @��
@�{���׾B�\�ٙ�C�o\@�{��Q�>�  @G�C�q�                                    By
�X  �          @�  @�����=q�aG�����C���@�����=q>aG�?�(�C���                                    By
�f�  �          @ָR@�\)��ff=���?W
=C���@�\)���?��@�
=C��                                     By
�u^  "          @�=q@�p����
>��R@%C�` @�p����?@  @ə�C���                                    By
��  �          @�Q�@����=q>��
@.{C��@����Q�?G�@�(�C�9�                                    By
���  �          @�
=@����N�R@S�
A�G�C�O\@����7�@hQ�B(�C���                                    By
��P  "          @��@�z��B�\@k�B33C�aH@�z��(��@~�RBC�0�                                    By
���  �          @�Q�@����P  @~{B�RC���@����4z�@���BQ�C��\                                    By
���  �          @�(�@����`  @h��Bp�C�]q@����Fff@~�RB��C�f                                    By
��B  �          @�  @��
�q�@1�A�33C��R@��
�^�R@I��A�ffC���                                    By
���  �          @ȣ�@�=q�l��@:�HA߮C��
@�=q�X��@Q�A�ffC��                                    By
��  �          @��@��
�s�
@I��A�\C���@��
�^{@aG�B��C��{                                    By
��4  �          @���@�p��j=q@L(�A�\)C�Y�@�p��Tz�@c33Bz�C��R                                    By
��  T          @�{@�Q��e@Mp�A�C��@�Q��P  @c33B33C�L�                                    By
��  �          @�{@����y��@J=qA���C��@����dz�@a�B�
C�P�                                    By
�%&  T          @�ff@�33�w
=@J=qA�z�C�^�@�33�a�@aG�BG�C���                                    By
�3�  "          @�
=@�33�]p�@e�B  C��=@�33�E@y��B��C�xR                                    By
�Br  �          @�G�@��H�g�@Q�A���C�)@��H�Q�@g
=BG�C�t{                                    By
�Q  
�          @Ӆ@�Q��^{@i��BQ�C�g�@�Q��G
=@~{BC���                                    By
�_�  T          @��@�z��^{@x��B�
C�H@�z��E@�ffBffC���                                    By
�nd  
�          @�(�@���\��@|(�BC���@���Dz�@�  BG�C�z�                                    By
�}
  
Z          @��
@�(��`��@u�B�C��3@�(��HQ�@�z�B�\C�ff                                    By
���  �          @��
@����\��@j=qBQ�C��@����E@}p�B33C�#�                                    By
��V  �          @�33@�=q�\(�@e�B�
C���@�=q�Fff@xQ�B��C�(�                                    By
���  �          @��H@�z��[�@s33B��C�.@�z��Dz�@�33B�C���                                    By
���  �          @�=q@�Q��^�R@w�B=qC���@�Q��G
=@��Bp�C��                                    By
��H            @ҏ\@x���\(�@�G�B�\C�G�@x���C�
@��\B$C���                                    By
���  �          @��@x���]p�@\)B��C�+�@x���E@���B#�C�                                    By
��  �          @�G�@}p��Y��@|(�BG�C���@}p��B�\@�\)B!
=C�E                                    By
��:  T          @У�@|���U@~�RB�C��f@|���>�R@�  B"�C�~�                                    By
� �  �          @��
@~{�AG�@}p�B�C�c�@~{�*=q@�ffB%G�C��                                    By
��  �          @�(�@�z��Tz�@a�Bz�C�� @�z��@��@s�
Bz�C��                                    By
�,  �          @�=q@����Tz�@QG�A���C�3@����A�@b�\B(�C�T{                                    By
�,�  �          @���@l���X��@b�\B��C��q@l���E�@tz�B�\C��                                    By
�;x  �          @ȣ�@w��=p�@|��B{C�B�@w��'�@�B'Q�C���                                    By
�J  �          @Ǯ@y���8��@z=qB�C��{@y���#�
@�z�B&p�C�\)                                    By
�X�  T          @�{@w
=�9��@w�B��C���@w
=�$z�@��HB%�RC�'�                                    By
�gj  T          @�
=@|(��1�@z=qB  C�c�@|(����@��
B&\)C��                                    By
�v  �          @�z�@��
�3�
@eB�C���@��
� ��@s33B��C�\)                                    By
���  �          @�G�@�ff�?\)@'�A�  C��@�ff�1G�@6ffA�  C��                                    By
��\  T          @��@�=q�O\)?�=qA(��C��@�=q�I��?��AP��C�o\                                    By
��  �          @�
=@�(��U?O\)@�  C��@�(��QG�?���A$��C��                                    By
���  �          @�ff@���S33>�@�p�C��@���P��?=p�@��C�:�                                    By
��N  �          @�{@�=q�]p�=u?�C�'�@�=q�\��>���@I��C�5�                                    By
���  �          @���@�Q��^{�u��
C���@�Q��^�R<�>�z�C��                                    By
�ܚ  �          @��@�Q��^{��Q��aG�C���@�Q��^�R���Ϳs33C���                                    By
��@  �          @�p�@�
=�c33=u?!G�C���@�
=�b�\>���@N{C��
                                    By
���  �          @��@�33�U�����:=qC��
@�33�W
=�L�;�C���                                    By
��  �          @�@����Z=q����ffC�O\@����Z=q=�?�33C�N                                    By
�2  �          @�@�\)�_\)�0����C���@�\)�a녾�(���{C���                                    By
�%�  �          @�{@��\�U��n{��C���@��\�X�ÿ.{��33C�w
                                    By
�4~  �          @�ff@���`�׿O\)����C�� @���c�
������\C��\                                    By
�C$  �          @�
=@����\�ͿTz����C�&f@����`  �z���33C��{                                    By
�Q�  �          @��R@����\�ͿQ���z�C�3@����`  ������C��                                    By
�`p  �          @�
=@����Z�H�n{��RC�J=@����^{�0����33C��                                    By
�o  �          @��R@�\)�b�\�=p����
C���@�\)�e���H���C�g�                                    By
�}�  �          @��@�  �hQ쾽p��a�C�N@�  �i������33C�=q                                    By
��b  �          @��@�
=�hQ�
=��z�C�7
@�
=�j=q��{�QG�C�R                                    By
��  �          @��@�
=�h�ÿ����HC�.@�
=�j�H��z��0  C�3                                    By
���  �          @�ff@�G��s�
��Q��\(�C���@�G��u����Ϳ�  C��\                                    By
��T  �          @�=q@�ff�s33��(���33C���@�ff�tz�8Q�޸RC�w
                                    By
���  �          @Å@�
=�`  �!G�����C�o\@�
=�a녾����p��C�O\                                    By
�ՠ  �          @��
@����h�þ���p�C��3@����i������=qC��=                                    By
��F  �          @���@����n{>\)?�=qC�e@����l��>\@aG�C�t{                                    By
���  T          @�p�@�ff�l(�>.{?���C��@�ff�j�H>���@p  C���                                    By
��  �          @��@����c�
>aG�@�
C�P�@����b�\>�G�@��C�e                                    By
�8  �          @�p�@��R�j=q>��R@5�C��f@��R�h��?�@�p�C�޸                                    By
��  �          @��@�p��l��>�=q@{C���@�p��j�H>��H@��C���                                    By
�-�  �          @ƸR@�p��q�>�?�
=C�=q@�p��p��>�33@QG�C�J=                                    By
�<*  �          @�{@�33�vff=�\)?��C��@�33�u>�z�@.{C��\                                    By
�J�  �          @ƸR@�p��qG�>.{?���C�J=@�p��p  >Ǯ@g
=C�Y�                                    By
�Yv  �          @���@��\�n{?
=@�Q�C�5�@��\�j�H?L��@���C�\)                                    By
�h  �          @�(�@��H�`��?:�H@�C��@��H�]p�?n{A��C�1�                                    By
�v�  �          @�\)@�(����H�Tz���ffC���@�(���(�������C���                                    By
��h  �          @Ǯ@���������R�8��C���@�����  ���
�333C�y�                                    By
��  �          @�  @�Q�����>W
=?��HC���@�Q���Q�>�G�@�Q�C��R                                    By
���  �          @�
=@�  ��  =#�
>ǮC��R@�  �\)>��@=qC�                                      By
��Z  �          @�\)@�Q��|��>�@�Q�C�(�@�Q��z�H?+�@�\)C�E                                    By
��   �          @�ff@�{�hQ�?Q�@�z�C��)@�{�e�?��\A33C��                                    By
�Φ  �          @Ǯ@���,(�@�
A�z�C�l�@���$z�@��A�G�C��=                                    By
��L  �          @ƸR@����<��@�A�G�C��H@����5@�A��C�Y�                                    By
���  �          @�p�@�\)�O\)?�
=A|��C�z�@�\)�I��?���A�G�C�ٚ                                    By
���  �          @�z�@��R�XQ�?��HA�\)C�.@��R�QG�@Q�A���C���                                    By
�	>  �          @�z�@����7�?�p�A���C�s3@����1�?�\)A�  C��
                                    By
��  �          @�(�@����@��?�z�AT��C��)@����;�?ǮAk�C�*=                                    By
�&�  �          @���@��
�Tz�?�G�A�\)C�� @��
�N�R?�A��
C�=q                                    By
�50  �          @���@����C33?��AJ=qC��\@����>�R?��RA`��C��R                                    By
�C�  �          @�(�@�  �9��?��A,z�C���@�  �5?��
AAp�C�Ǯ                                    By
�R|  T          @�p�@����C�
?uA  C��\@����@��?�{A%��C�"�                                    By
�a"  �          @��
@�z��G�?�\)A(Q�C�]q@�z��C�
?�G�A>=qC��
                                    By
�o�  �          @��
@��\�>�R?У�Aw\)C�˅@��\�:=q?�\A�=qC��                                    By
�~n  �          @�(�@����HQ�?�=qAn�\C�f@����C�
?�(�A�(�C�U�                                    By
��  �          @���@�Q��H��?�z�Az�RC���@�Q��Dz�?�ffA�{C�C�                                    By
���  �          @�z�@��\�C33?�{As\)C�}q@��\�>�R?޸RA�  C�˅                                    By
��`  �          @�z�@�Q��K�?�p�A`(�C�Ǯ@�Q��G�?�\)AuG�C�                                    By
��  �          @�@�(��:=q@�A���C��q@�(��333@#33A���C�{                                    By
�Ǭ  �          @�@�\)�<(�@
=qA�{C��R@�\)�6ff@�\A�C�                                      By
��R  �          @�@��
�>{@�A��\C�J=@��
�7�@\)A�Q�C���                                   By
���  �          @�
=@����@  @A�
=C���@����:=q@p�A��\C���                                   By
��  �          @�ff@��R�O\)?�A�z�C�k�@��R�J=q?�(�A�z�C��)                                    By
�D  �          @ƸR@�G��I��?�\A��\C��q@�G��E�?��A�{C�J=                                    By
��  �          @ƸR@���Mp�?�=qA�G�C��R@���H��?��HA���C��f                                    By
��  �          @Ǯ@���X��?�ffA�{C��3@���Tz�?�
=A�  C��)                                    By
�.6  �          @�G�@�z��[�?��RA��C�w
@�z��W
=@
=A�
=C��                                    By
�<�  �          @ƸR@����Q�?�\)ArffC�l�@����Mp�?޸RA�z�C���                                    By
�K�  �          @�ff@�{�Y��?ǮAiG�C�@�{�U?�
=A|Q�C���                                    By
�Z(  �          @�  @��_\)?���Am�C�U�@��[�?�p�A�z�C���                                    By
�h�  �          @�
=@�\)���@A�
=C��=@�\)�z�@�A�C��                                    By
�wt  �          @�
=@�G��Mp�?ٙ�A~ffC��q@�G��I��?�A�C���                                    By
��  �          @�
=@��
�`��?��Au�C�  @��
�\��?�G�A�C�Y�                                    By
���  �          @�ff@���
�H?�(�A�  C�7
@���
=?��A��C�|)                                    By
��f  �          @�ff@�����?���Aj�\C��{@����G�?�33AvffC�Ф                                    By
��  �          @�@�Q��/\)?�(�A���C�@ @�Q��+�?�A�z�C��                                     By
���  �          @ƸR@����8��?���A�ffC�Z�@����4z�@�\A��C��H                                    By
��X  �          @�\)@��R�3�
?���A�C���@��R�0  @�\A���C�
                                    By
���  �          @�ff@���<(�?��
A�\)C�*=@���8Q�?�\)A�Q�C�ff                                    By
��  �          @�@��8��?��HA���C�h�@��5?�ffA��C���                                    By
��J  �          @�@�{�7
=?�p�A�ffC��{@�{�3�
?���A���C��                                   By
�	�  �          @�@��8Q�?�(�A�C�w
@��5�?�A�Q�C��\                                    By
��  �          @�p�@�\)�<��?��AP��C�AH@�\)�:=q?�(�A]��C�n                                    By
�'<  �          @���@��
�0��?��A*�HC�h�@��
�.�R?�(�A6�RC���                                    By
�5�  �          @�z�@�p��,(�?��A��C���@�p��*=q?�\)A'�
C��                                    By
�D�  �          @�(�@�ff�0��?��HA�p�C�
=@�ff�-p�?��
A�G�C�>�                                    By
�S.  �          @�p�@�(��@  @33A���C�,�@�(��<(�@Q�A�\)C�q�                                    By
�a�  �          @���@��@  @Q�A�C�XR@��<(�@��A�  C��
                                    By
�pz  �          @��@�
=�;�@
=qA��
C��H@�
=�7�@�RA�C�                                      By
�   �          @�z�@�G��Dz�?У�Aw�C�L�@�G��A�?��HA�C�y�                                    By
���  �          @�p�@�����H@�A��RC�c�@����
=@�HA�p�C��=                                   By
��l  �          @��@�z��=q@=qA��C�o\@�z��ff@p�A��\C��{                                   By
��  �          @���@�33�Q�@�RA�=qC�� @�33�z�@"�\A���C��f                                    By
���  �          @��@�(��
=@+�A�ffC��@�(��33@.�RA�ffC��                                    By
��^  �          @Å@�33� ��@,��Aљ�C�C�@�33����@/\)A�\)C���                                    By
��  �          @��R@����&ff?�A�C�U�@����#�
?�\)A�ffC��                                    By
��  �          @��R@�G��%?�ffA��\C�c�@�G��#33?���A��C���                                    By
��P  �          @��@�  �%@G�A��C�T{@�  �"�\@z�A�{C��f                                    By
��  �          @�@�{�4z�?�
=A��
C�!H@�{�2�\?޸RA�z�C�H�                                    By
��  �          @�{@�(��333?�33A�p�C�
@�(��0��?��HA�  C�C�                                    By
� B  �          @��R@����@  ?�=qAv�\C�C�@����>{?��A�
C�ff                                    By
�.�  T          @���@�z��7
=?�33A�  C��3@�z��5�?��HA�ffC��
                                    By
�=�  �          @��@�\)�%�?��A�  C�G�@�\)�#33?�A��C�p�                                    By
�L4  �          @�\)@��R�(��@�A�33C�  @��R�&ff@�A��C�+�                                   By
�Z�  �          @���@��\�'�@{A�C���@��\�%�@!G�Ař�C��
                                   By
�i�  �          @�\)@�p��>{@'
=AУ�C�@�p��;�@*=qA�
=C�B�                                    By
�x&  �          @���@�=q�aG�?#�
@�(�C���@�=q�`  ?333@�{C�H                                    By
���  �          @���@�\)�p  >��R@>{C�˅@�\)�p  >�p�@c33C�Ф                                    By
��r  �          @�Q�@�p��qG������J=qC��\@�p��qG���=q�%C���                                    By
��  �          @���@����g��333��C�|)@����hQ�&ff���C�s3                                    By
���  �          @�  @���c�
�p�����C���@���dz�c�
�z�C���                                    By
��d  �          @�p�@��W
=���\�G�C�.@��W���p��@  C�)                                    By
��
  �          @�  @���i���s33�=qC���@���j=q�fff�
ffC���                                    By
�ް  �          @�Q�@�p��l(��Y����\C��)@�p��l�ͿL����{C�Ф                                    By
��V  �          @�\)@����a녿0����(�C��q@����b�\�#�
��{C��{                                    By
���  �          @��
@�G��Tz�8Q���Q�C��@�G��U��+���33C��)                                    By
�
�  �          @��
@��
�L(��5��{C�b�@��
�L�Ϳ+����C�Y�                                    By
�H  �          @��H@��R�@  �5��
=C�e@��R�@�׿+���(�C�\)                                    By
�'�  �          @��@�z��C33�8Q���33C�f@�z��C�
�.{��Q�C��q                                    By
�6�  �          @���@���E��E�����C��{@���E�:�H��=qC���                                    By
�E:  �          @�=q@��\�J=q�:�H��RC�` @��\�J�H�333��z�C�XR                                    By
�S�  �          @�(�@��
�Mp��+���33C�O\@��
�N{�#�
����C�H�                                    By
�b�  �          @��\@����P�׿=p���Q�C��H@����P�׿5��ffC���                                    By
�q,  �          @��@�ff�Tz�B�\��C�b�@�ff�U��:�H��  C�\)                                    By
��  �          @�
=@���W
=�W
=�ffC��\@���W��O\)�C�Ǯ                                    By
��x  �          @�@�Q��X�ÿ���\C��=@�Q��Y������G�C��                                    By
��  �          @���@����c�
���H��=qC��
@����dz����G�C��3                                    By
���  �          @�ff@�
=�`�׾�����Q�C��\@�
=�`�׾�p��p  C���                                    By
��j  �          @��R@�  �`  ��z��8Q�C�@�  �`�׾���(Q�C��                                    By
��  �          @�
=@���l(��k���\C��f@���l(��L���33C��                                    By
�׶  �          @�p�@�=q�j�H�#�
���C��\@�=q�k��\)��z�C��                                    By
��\  �          @���@����n{��\)�1�C�� @����n{����$z�C�޸                                    By
��  �          @��H@��\�g������G�C���@��\�g�������C�ٚ                                    By
��  �          @�p�@����]p�=�Q�?W
=C�
@����]p�=���?�G�C�R                                    By
�N  �          @�{@��H��G��#�
��=qC���@��H��G��\)��33C���                                    By
� �  �          @�(�@��H���R<#�
=�C��@��H���R<�>���C��                                    By
�/�  �          @��H@��
��z�=�G�?�=qC�ff@��
��z�>�?�p�C�g�                                    By
�>@  �          @�=q@�G���������C�{@�G��������33C�3                                    By
�L�  �          @��@�  ����>.{?ٙ�C�K�@�  ����>B�\?�=qC�L�                                    By
�[�  �          @��H@�Q��\)>L��?��HC�j=@�Q��\)>W
=@�
C�k�                                    By
�j2  �          @�33@���=q>�z�@3�
C���@���=q>���@9��C���                                    By
�x�  �          @��H@��
��(�=�?�(�C�z�@��
��(�>�?�ffC�z�                                    By
��~  �          @�G�@�������=�?�Q�C�@�������>�?�G�C�                                    By
��$  �          @��@�33���>k�@�
C�q�@�33���>u@
=C�q�                                    By
���  �          @�=q@������>�@�Q�C��@������>�@���C��                                    By
��p  �          @���@~{��>\@s33C��H@~{��>Ǯ@tz�C��H                                    By
��  �          @�  @�Q����H>�
=@��C�7
@�Q����H>�
=@��C�7
                                    By
�м  �          @���@�
=�{�>\@p��C���@�
=�{�>\@p��C���                                    By
��b  �          @��H@~�R��  ?��@��C��@~�R��  ?��@�z�C��                                    By
��  �          @��@������>�(�@�z�C�@ @������>�
=@�33C�@                                     By
���  �          @�z�@��H�~{>�=q@&ffC��H@��H�~{>��@#33C��H                                    By
�T  �          @���@�  ���\>��@!G�C�
@�  ���\>�  @p�C��                                    By
��  T          @��@�����G�>u@�
C�` @�����G�>k�@\)C�`                                     By
�(�  �          @�z�@��\�~�R>��
@Dz�C���@��\�~�R>��R@>�RC��R                                    By
�7F  �          @���@����z=q>�Q�@`��C�(�@����z�H>�33@Z=qC�'�                                    By
�E�  �          @��@��H�z=q>��@#33C���@��H�z=q>�  @(�C��q                                    By
�T�  �          @�=q@��R��Q���Ϳ�  C�4{@��R��Q��G���\)C�4{                                    By
�c8  �          @�@|�����?uA�C�>�@|�����?p��A�RC�<)                                    By
�q�  �          @��@~{����?Q�@��
C��@~{���?L��@��RC�f                                    By
���  �          @�  @�p���Q�?+�@��C�0�@�p���Q�?&ff@ǮC�.                                    By
��*  �          @�{@�Q����>��
@E�C��@�Q����>���@9��C�H                                    By
���  �          @�ff@�{�z�H>���@K�C�G�@�{�z�H>��R@@  C�Ff                                    By
��v  �          @�p�@��
�~�R>�\)@0  C���@��
�~�R>��@#33C�ٚ                                    By
��  �          @��@�p���Q�?��@�G�C���@�p���Q�?�@��\C��H                                    By
���  T          @��R@���qG�?+�@�\)C�:�@���qG�?&ff@ȣ�C�7
                                    By
��h  �          @�\)@����p��>L��?�33C���@����p��>8Q�?�
=C���                                    By
��  �          @�Q�@���a녽�\)�+�C�  @���a녽�Q�aG�C�                                      By
���  �          @���@���W
=�����z�C��@���W
=�.{��\)C�
=                                    By
�Z  �          @�G�@����U�=���?k�C�Ff@����U�=�\)?333C�E                                    By
�   �          @���@�p��\��=�\)?5C�~�@�p��\��=L��>�C�~�                                    By
�!�  �          @���@�{�Z=q>�p�@b�\C���@�{�Z�H>�33@S33C���                                    By
�0L  �          @��@���Z=q>�\)@,��C���@���Z=q>��@��C��)                                    By
�>�  �          @�  @�(��XQ�?Y��AffC��@�(��XQ�?Q�@�z�C���                                    By
�M�  �          @�\)@���n{>�\)@+�C���@���n{>�  @Q�C��                                    By
�\>  �          @���@���mp�?:�H@�\)C��)@���mp�?333@�C���                                    By
�j�  �          @�33@�z��s33?��@��C���@�z��s�
?�@��\C��                                    By
�y�  �          @�=q@�G��x��?�@�33C�޸@�G��y��>�@�  C���                                    By
��0  �          @�z�@�����>�G�@��\C�)@�����>��@}p�C��                                    By
���  �          @�p�@�����  ?#�
@�ffC��@�����Q�?��@��\C��                                     By
��|  �          @���@�����{�#�
��G�C�Ф@�����{�L�Ϳ�33C���                                    By
��"  �          @���@�(��������\��C��@�(���G�����"=qC��
                                    By
���  �          @���@�G��p  �k���C�<)@�G��o\)��=q�)��C�>�                                    By
��n  �          @�=q@��
�fff>��@   C�Ǯ@��
�fff>k�@	��C��f                                   By
��  �          @Å@�33�l��=u?\)C�S3@�33�l��<��
>B�\C�S3                                   By
��  �          @�ff@�p��k�����(�C���@�p��k��.{����C��                                    By
��`  �          @��@�\)�qG���Q�\(�C��q@�\)�qG�����  C���                                    By
�  �          @�Q�@�ff�o\)=�Q�?c�
C��q@�ff�o\)=L��>�C��q                                    By
��  �          @���@���z=q>#�
?���C�0�@���z�H=�?���C�/\                                    By
�)R  �          @�  @�{�l�ͼ����
C���@�{�l�ͽ��
�=p�C��)                                    By
�7�  �          @�G�@��
�u    �#�
C�  @��
�u�L�;��C�                                      By
�F�  �          @\@�(���ff>�z�@-p�C�q@�(���ff>u@�RC��                                    By
�UD  �          @�=q@���p  >�G�@��C���@���p��>Ǯ@n{C���                                    By
�c�  �          @���@���I��?(��@��C�(�@���J=q?�R@�C�                                      By
�r�  �          @�=q@�G��U?:�H@��HC�5�@�G��Vff?.{@�{C�+�                                    By
��6  �          @���@��H�K�>���@r�\C��@��H�L(�>�Q�@Y��C��q                                    By
���  �          @��@���Fff?
=q@�(�C���@���Fff?   @��C�z�                                    By
���  T          @���@���[�>�  @z�C��R@���\(�>L��?��C��{                                    By
��(  �          @�\)@���a�>���@6ffC���@���a�>�  @��C��
                                    By
���  �          @�{@�p��n�R?\)@�p�C�l�@�p��n�R?�\@�ffC�e                                    By
��t  �          @��
@����j=q?L��@�\C�S3@����k�?@  @�33C�H�                                    By
��  �          @\@���R�\?��A&�\C�)@���S33?��A\)C��                                    By
���  �          @�Q�@�{�g
=?@  @�C�8R@�{�g�?0��@�{C�/\                                    By
��f  �          @���@�G��g�?.{@�ffC�w
@�G��hQ�?!G�@��RC�n                                   By
�  �          @�33@����n{�L�Ϳ�
=C�
@����n{����(�C��                                   By
��  �          @�33@�(��e��0����G�C��@�(��dz�@  ��G�C���                                    By
�"X  z          @��@�\)�Vff�O\)��{C�H@�\)�U�\(���RC�                                    By
�0�  .          @�=q@���R�\����T(�C�{@���QG���Q��[�
C�*=                                    By
�?�  �          @�z�@�ff�Vff��z��T��C��R@�ff�TzῺ�H�\Q�C�                                    By
�NJ  �          @�33@�
=�J=q�����n=qC���@�
=�H�ÿ�{�u��C��                                    By
�\�  �          @Å@��\�]p������Hz�C�33@��\�\(���\)�P��C�H�                                    By
�k�  �          @���@�z��2�\����ffC�(�@�z��0  �����C�N                                    By
�z<  �          @�p�@��\�+��&ff�У�C��3@��\�(���(����=qC��                                    By
���  �          @�p�@n{�a��<(���\C�H�@n{�^�R�@  ��C�y�                                    By
���  �          @���@�\)�@  �����p�C�)@�\)�=p��(���p�C�G�                                    By
��.  �          @��H@��R�%��G���G�C��@��R�"�\��
����C�Ф                                    By
���  �          @���@��\�N�R�J=q� z�C�g�@��\�Mp��W
=�	�C�w
                                    By
��z  �          @�z�@�33�'�@*�HAθRC���@�33�*�H@(Q�A��C���                                    By
��   �          @�z�@�\)��@r�\B
=C�� @�\)�\)@p  Bp�C�P�                                    By
���  �          @��@��\�
=q@j�HB�C��R@��\�{@h��B	�\C���                                    By
��l  �          @�=q@�=q�1�@#33A�(�C���@�=q�4z�@   A�z�C�t{                                    By
��  �          @�
=@�  ��\@^{B
=C���@�  �ff@[�A��HC�e                                    By
��  �          @�
=@�z���@hQ�B��C���@�z��\)@fffB
=C���                                    By
�^  �          @���@��R�z�@j=qB�RC��@��R�Q�@g�B	��C���                                    By
�*  �          @��H@�z��Q�@fffB
�\C�h�@�z��(�@c�
BC�)                                    By
�8�  �          @ə�@��=q@]p�BG�C�` @��{@Z�HBp�C�
                                    By
�GP  �          @�G�@���Q�@Y��B��C���@���(�@VffB �
C�c�                                    By
�U�  �          @�  @�Q��\)@W�BffC�e@�Q��33@U�B �C�)                                    By
�d�  �          @���@���	��@VffB{C��@�����@S�
A��HC�˅                                    By
�sB  �          @�z�@�\)�
=@X��B p�C���@�\)�
�H@W
=A��C�=q                                    By
���  �          @�Q�@�{�@N{A��
C��@�{�	��@K�A���C�=q                                    By
���  �          @�  @�ff�
=q@K�A�C�=q@�ff�p�@H��A�z�C��
                                    By
��4  �          @ʏ\@����Q�@K�A�\)C���@���   @I��A�z�C�XR                                    By
���  �          @˅@�Q��
=q@P��A�
=C�T{@�Q��{@N{A�C�                                    By
���  �          @ʏ\@�����@Mp�A�C�!H@���  @J�HA�Q�C���                                    By
��&  �          @���@�Q��p�@HQ�A�ffC���@�Q�� ��@EA�RC���                                    By
���  �          @�Q�@�ff� ��@@  A�  C�!H@�ff�#�
@=p�A�Q�C���                                    By
��r  T          @���@�  �!G�@G
=A�Q�C���@�  �$z�@C�
A�\C�b�                                    By
��  �          @�G�@��R�"�\@9��A�{C�u�@��R�%@7
=A�=qC�9�                                    By
��  T          @�  @�
=�#�
@3�
A�33C�e@�
=�'
=@0��A�\)C�(�                                    By
�d  T          @���@�ff�0  @,��A�=qC�y�@�ff�333@)��A�{C�AH                                    By
�#
  �          @�  @�z��1�@-p�A��C�1�@�z��5�@*=qA�C���                                    By
�1�  �          @�Q�@���!�@3�
A�p�C���@���%�@0��Aљ�C�U�                                    By
�@V  �          @ə�@���33@�G�B��C�  @���(�@���BC��3                                    By
�N�  �          @��@��ÿ��@s33B�RC��@��ÿ�{@q�B�\C���                                    By
�]�  �          @Ǯ@�(��:=q@\)A��C��)@�(��<��@�A��RC�j=                                    By
�lH  �          @��H@����e?�z�A���C��@����g�?�A�G�C���                                    By
�z�  �          @��@�z��^�R@��A�p�C��\@�z��aG�@�A�(�C��f                                    By
���  �          @�@�33��(�@��
B'�RC���@�33��ff@�33B&Q�C�1�                                    By
��:  �          @�
=@��
����@�z�B(p�C�4{@��
���H@��
B'�C��H                                    By
���  �          @ə�@�����{@|��BffC��@�����
=@z�HBC��H                                    By
���  �          @ȣ�@����33@���B!  C��@����(�@�Q�B�C���                                    By
��,  �          @��H@�p��\@\)BQ�C���@�p��˅@}p�B  C���                                    By
���  T          @�
=@�=q�޸R@u�B{C���@�=q��@r�\B�C�.                                    By
��x  �          @�@�G���(�@s�
Bz�C���@�G����@qG�B��C�>�                                    By
��  �          @�\)@����
=@q�BQ�C��@����  @p  B�
C��{                                    By
���  �          @�@�z���@q�BC���@�z��{@p  BffC�j=                                    By
�j  �          @��H@�=q�ٙ�@h��B
=C���@�=q��G�@g
=B�C�s3                                    By
�  �          @��
@�=q��z�@mp�Bp�C��@�=q��(�@k�B��C���                                    By
�*�  �          @���@���\@p  B�
C�޸@���˅@n{Bz�C�z�                                    By
�9\  �          @���@����@p��B�C�ٚ@����z�@n�RBz�C�y�                                    By
�H  �          @��@�G��.{@+�A�Q�C���@�G��1G�@(Q�A�C���                                    By
�V�  �          @�p�@\�����H����C�\@\�����\��R����C��                                    By
�eN  �          @���@j=q���H�u���C�ٚ@j=q���\�����C33C��q                                    By
�s�  �          @�@s33��=q�E�����C�)@s33��녿\(���\)C�&f                                    By
���  �          @ƸR@�����\���H��=qC�  @����=q�z���33C�(�                                    By
��@  �          @ƸR@��\��p�=u?�C���@��\��p�����=qC���                                    By
���  �          @�
=@�(�����>.{?�ffC��{@�(�����=��
?B�\C��3                                   By
���  �          @�{@����p�    �#�
C���@����p���Q�Tz�C���                                   By
��2  �          @�p�@y������>\)?��\C���@y������=L��>�C��q                                    By
���  �          @��H@i��������Ϳh��C�s3@i������8Q�޸RC�t{                                    By
��~  �          @��H@u����?#�
@���C���@u���p�?��@�
=C���                                    By
��$  �          @���@xQ����?c�
A��C�n@xQ���  ?O\)@�Q�C�b�                                    By
���  �          @��
@Tz���������C�|)@Tz���33�\)��G�C���                                    By
�p  �          @��H@L�����
�+���33C��)@L������B�\��
=C��                                    By
�  �          @�(�@k���=q���Ϳp��C��\@k���=q�B�\��G�C���                                    By
�#�  �          @�p�@�G������Q�^�RC��H@�G�����.{�У�C���                                    By
�2b  �          @�z�@l(���33>\@fffC��f@l(���33>���@2�\C��H                                    By
�A  T          @��
@tz���ff��  �ffC��@tz���ff�����HQ�C���                                    By
�O�  �          @���@a���(�?��HA5C��{@a�����?�\)A(��C��                                    By
�^T  �          @�(�@aG����R?&ff@��C���@aG���
=?\)@��C��                                     By
�l�  �          @���@�������?��A�HC���@�������?z�HA
=C���                                    By
�{�  
�          @�(�@���p��?�
=AX(�C�c�@���q�?�{AN=qC�L�                                    By
��F  
Z          @�(�@�G��tz�?�(�A^{C���@�G��u?�33AT  C��H                                    By
���  
�          @��@����w�?�  Aa�C���@����y��?�
=AW�C���                                    By
���  "          @�33@}p���33>.{?˅C�]q@}p���33=�Q�?Y��C�\)                                    By
��8  �          @\@qG���
=?��@��C�S3@qG���\)>��@���C�K�                                    By
���  "          @��
@N{����?�33A��C��f@N{���\?���A��C�Ф                                    By
�ӄ  �          @�(�@U���?�33AV�\C�g�@U���\?���AI�C�XR                                    By
��*  
�          @���@g�����?��RA:�HC��f@g���=q?�z�A.�RC�xR                                    By
���  
Z          @�{@�����R?�{AK�C�/\@����\)?��A@z�C�q                                    By
��v  
�          @�@����G�?�Q�A2=qC��)@�����?�\)A(Q�C��                                    By
�  "          @�z�@�{���\?���A*�\C���@�{��33?���A Q�C���                                    By
��  
Z          @��
@����~�R?}p�Ap�C�Q�@�����  ?n{A�C�C�                                    By
�+h  
�          @�(�@��
�e�?��
A\)C��{@��
�fff?z�HA�\C��                                    By
�:  T          @�z�@�\)�^{?��Ay�C�ٚ@�\)�`  ?˅AqG�C��                                     By
�H�  T          @���@���l��@�A�Q�C��3@���n�R@G�A��C��\                                    By
�WZ  T          @��H@�  �\)?�\)Ax(�C�ff@�  ��Q�?�ffAnffC�P�                                    By
�f   
�          @���@�33���\?333@ָRC���@�33���H?!G�@��C���                                    By
�t�  
�          @\@�  ����?�Q�A]�C�P�@�  ��G�?���AS�C�<)                                    By
��L  T          @���@��R�_\)@�\A��\C���@��R�aG�?��RA�=qC�ٚ                                    By
���  �          @�Q�@��
�XQ�@ffA�ffC�#�@��
�Z�H@33A�(�C���                                    By
���  
(          @�  @�Q��[�?��HA��\C�aH@�Q��]p�?�z�A�ffC�C�                                    By
��>  �          @�
=@����l(�?��HA��HC��q@����n{?�33A�ffC��H                                    By
���  
�          @�  @����`  ?���A��
C��@����a�?�ffA�C�                                      By
�̊  "          @�G�@����`��?�=qAs�C�xR@����a�?��
Ak�C�aH                                    By
��0  �          @�=q@��\�h��?�{Av{C��=@��\�j=q?ǮAm�C��{                                    By
���  	�          @��@�\)�[�?˅Ar�HC��@�\)�\��?��Ak\)C���                                    By
��|  "          @���@��\�S33?��
Aj�HC�ٚ@��\�Tz�?��RAc�C���                                    By
�"  T          @\@�  �mp�?��HA�z�C�<)@�  �n�R?�33A|��C�%                                    By
��  �          @���@�Q��P��?��
A���C���@�Q��R�\?�p�A�G�C��\                                    By
�$n  �          @�\)@�33�L��?���AU��C�J=@�33�N{?�=qAN�RC�7
                                    By
�3  �          @�  @�p��H��?�=qAM��C���@�p��J=q?��AG33C��f                                    By
�A�  
�          @�  @�(��P  ?�ffAG�C�(�@�(��QG�?�  A@��C�R                                    By
�P`  T          @��@��1�?���A.=qC��f@��2�\?���A(��C���                                    By
�_  
�          @�Q�@�{�1G�?�A4z�C���@�{�1�?��A/
=C��f                                    By
�m�  T          @�
=@����5�?z�HA�RC���@����5?p��AG�C���                                    By
�|R  �          @��@��
�-p�?=p�@�  C�� @��
�-p�?333@�C���                                    By
���  �          @��
@���1�?�G�A\)C�E@���333?xQ�A=qC�8R                                    By
���  T          @�(�@����(Q�?�Q�A2�\C�H@����)��?�33A-C���                                    By
��D  "          @�33@�\)�*=q?�\)Av=qC�� @�\)�+�?�=qAqp�C�j=                                    By
���  
�          @Å@��\�7
=?�p�A�Q�C�G�@��\�8Q�?ٙ�A�C�1�                                    By
�Ő  "          @�z�@�=q�S�
?�=qA���C���@�=q�U�?��A��C��\                                    By
��6  �          @�@���_\)?��
AeG�C�(�@���`  ?��RA_�C��                                    By
���  "          @�p�@�z��XQ�?�\)As�C���@�z��X��?�=qAm�C��                                     By
��  �          @��H@����X��?�
=A�=qC�Ff@����Z=q?�33Az�HC�4{                                    By
� (  
4          @�z�@�G��]p�?�A|��C�3@�G��^�R?��Aw33C��                                    By
��  �          @���@���G
=?��A��C�
=@���HQ�?�  A���C��
                                    By
�t  
�          @��@��\�>{?�ffA�  C��R@��\�?\)?�\A��C��f                                    By
�,  T          @�33@����Dz�?�ffAj�RC�Y�@����E�?\Af=qC�J=                                    