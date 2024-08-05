CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240301000000_e20240301235959_p20240302020550_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-03-02T02:05:50.133Z   date_calibration_data_updated         2024-01-10T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-03-01T00:00:00.000Z   time_coverage_end         2024-03-01T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�t�@            @����xQ��(�����  CQh��xQ�!G��J=q�33C=G�                                    Bx�u�  
o          @�G��tz��
�H�=q��ffCQ���tz����L(��(�C<�)                                   Bx�u�  �          @�Q��{����(����
CQ��{��8Q��@�����C>p�                                    Bx�u$2  �          @��R�q��
=q�
=��CQ�3�q녿�R�H���33C==q                                    Bx�u2�  �          @�
=�l���p��{��z�CR�f�l�Ϳ(��P���${C=O\                                    Bx�uA~  �          @�ff�i���33�=q��CTE�i���5�P���$��C?�                                    Bx�uP$  "          @��R�w
=��
����z�CP(��w
=����C�
�C<!H                                    Bx�u^�  T          @��u���\�����CP��u����C�
���C;ٚ                                   Bx�ump  �          @�{�|(�����Q���\CLs3�|(���z��=p���\C8B�                                   Bx�u|  �          @��q�� �������\CP��q녾�ff�H���G�C:�q                                    Bx�u��  �          @�Q��u�� ���!G���ffCO���u������L�����C:�                                    Bx�u�b  
�          @�  �p  � ���(Q����CPB��p  ��Q��R�\�$��C9u�                                    Bx�u�  
w          @�\)�o\)�  �����Q�CS{�o\)�.{�N{�!�C>T{                                    Bx�u��            @�Q��p  �z��%�����CP���p  ��(��Q��$  C:��                                    Bx�u�T  T          @���u�33�$z����RCP\�u��
=�P��� �C:.                                    Bx�u��  "          @����u���%���p�CO�
�u�����P��� �C9�                                    Bx�u�  �          @���vff��� ����CP�H�vff���O\)�z�C;��                                    Bx�u�F  T          @����xQ���{��z�CPJ=�xQ�   �L���
=C;ff                                    Bx�u��  
          @�G��w��������CP��w�����K��z�C<
                                    Bx�v�  "          @�  �z�H�G��ff���COW
�z�H��\�C�
�{C;xR                                    Bx�v8  
�          @�(��w���p��p���p�CO\�w�����:=q�C<�                                    Bx�v+�  
�          @�z��p  �(��\)��ffCRB��p  �5�C33��C>�                                     Bx�v:�  �          @�{�p�������хCS�)�p�׿^�R�Dz��{CA{                                    Bx�vI*  T          @�{�qG��#�
��z���ffCV8R�qG������=p��p�CE�3                                    Bx�vW�  �          @��qG��(���
=���CT޸�qG�����:=q�33CD)                                    Bx�vfv  
�          @�\)�w��
�H�
�H�У�CQB��w��:�H�>�R�p�C>�q                                    Bx�vu  
�          @��
�u�"�\�������\CU�=�u��\)�*=q�z�CG��                                    Bx�v��  	�          @����u��-p������\CWT{�u��˅�'
=�  CJ��                                    Bx�v�h  T          @��\�s�
�*=q��ff�z�RCV���s�
��{�\)���\CJ��                                    Bx�v�  
(          @�G��g
=�\)�
=q��G�CS�f�g
=�O\)�@  �p�C@��                                    Bx�v��  T          @�33�`���z��'���CR��`�׾�
=�Tz��,�C:ٚ                                    Bx�v�Z  �          @�\)�dz��
=�#�
��
CMB��dz���C�
�"ffC5��                                    Bx�v�   	�          @Tz��{���R��G�����CS0��{�\�G��)\)C<�f                                    Bx�vۦ  	�          @[��Q�E��  �((�CE�3�Q�?   ���/�C(�                                    Bx�v�L  	�          @e��p���  �����CWT{�p���\�"�\�5=qC?��                                    Bx�v��  
�          @����R�\�!G������=qCYc��R�\�k��Vff�1��CC�                                    Bx�w�  
�          @��\�Tz����'��ffCW��Tz�&ff�\(��5��C?�                                    Bx�w>  T          @�33�W���Q��8���(�CQ��W��8Q��^{�7(�C7{                                    Bx�w$�  
�          @�Q��W���\�*=q�G�CS.�W��\�U��2
=C:k�                                    Bx�w3�  "          @���\�Ϳ�G��>{��CK�{�\��>k��Tz��/ffC0)                                    Bx�wB0  
�          @��H�\�Ϳ����H���!z�CH�f�\��>��W��0=qC,
=                                    Bx�wP�  "          @����Y������N{�(�
CD��Y��?E��S33�-�C'J=                                    Bx�w_|  
�          @��\�Y����33�N�R�'�HCF���Y��?+��W
=�0�\C(޸                                    Bx�wn"  
�          @��\�Z�H��  �K��'  CDaH�Z�H?E��O\)�+=qC'Q�                                    Bx�w|�  
�          @���_\)�ٙ��?\)�z�CM���_\)=��
�\(��233C2�R                                    Bx�w�n  
�          @�{�dz��Q��<���{CMY��dz�=u�Y���.p�C3�                                    Bx�w�  �          @�ff�\�ͿУ��HQ����CME�\��>W
=�a��6G�C0�{                                    Bx�w��  "          @��W
=���E���CP� �W
=    �fff�;�HC4                                      Bx�w�`  T          @�{�\�Ϳ޸R�C�
��CN�q�\��=�\)�aG��633C2��                                    Bx�w�  
(          @���Q����<(����CU��Q녾�p��h���?ffC:}q                                    Bx�wԬ  �          @�{�W��
=�<(���CT{�W����R�g
=�;G�C9J=                                    Bx�w�R  �          @���w
=��33�5��
{CG޸�w
=>u�H���G�C0ff                                    Bx�w��  
�          @�Q��j�H���H�<���ffCL�R�j�H=#�
�Z=q�+��C3h�                                    Bx�x �  	�          @�  �l(���=q�?\)��RCK(��l(�>.{�W��)�RC1c�                                    Bx�xD  i          @�
=�]p���z��AG���\CP�H�]p������dz��7�\C5��                                    Bx�x�  "          @�
=�^�R����@  ���CP���^�R�����c33�6{C5��                                    Bx�x,�  T          @�
=�U���33�J�H�p�CQ�3�U����
�l���?��C4L�                                    Bx�x;6  �          @�
=�R�\��
=�K��\)CRY��R�\�#�
�n{�B  C4�f                                    Bx�xI�  
�          @��R�L(�����R�\�&�CR���L(�=#�
�r�\�G��C38R                                    Bx�xX�  �          @�
=�O\)��z��P  �#\)CR���O\)<#�
�qG��Ep�C3޸                                    Bx�xg(  T          @�p��H���G��N{�#{CT� �H�ý����s33�I��C5�)                                    Bx�xu�  
�          @���A���
�P���&��CV&f�A녽�G��w
=�O�C6(�                                    Bx�x�t  �          @�(��HQ��33�Mp��%33CS:��HQ�    �n�R�H�C4                                    Bx�x�  �          @�
=�I����\�X���,��CQG��I��>L���tz��I�C0Y�                                    Bx�x��  T          @�  �>{��ff�l���?G�CO���>{?
=q�~{�S{C)��                                    Bx�x�f  �          @����:�H�����x���IG�CMB��:�H?J=q�����U{C$�                                    Bx�x�  T          @����Mp���z��S33�%�HCRǮ�Mp�<��s�
�G�RC3�=                                    Bx�xͲ  
�          @���E�
=q�J=q� (�CV��E����tz��K�
C8�q                                    Bx�x�X  T          @���K���
�HQ��p�CU
=�K��B�\�p  �F�RC7}q                                    Bx�x��  "          @�z��<(���R�Mp��#�CY��<(������x���S{C9��                                    Bx�x��  r          @���Fff�z��G
=�G�CU�=�Fff�aG��n�R�H�HC8                                      Bx�yJ  �          @��
�S�
��=q�E��p�CP�R�S�
���e��<��C4u�                                    Bx�y�  �          @����Z�H��  �C33���CO#��Z�H<��aG��7ffC3�                                     Bx�y%�  
�          @�33��R���W��0��C`����R������z��k(�C=(�                                    Bx�y4<  6          @���p��'
=�^{�5��Ce� �p��   ����y{C@ٚ                                    Bx�yB�  
�          @�  ���  �XQ��6ffC^� ���u�����k�C9��                                    Bx�yQ�  "          @�Q��k��333�;��=qC>���k�?^�R�8���z�C&��                                    Bx�y`.  �          @����p  �8Q��7
=�\)C>�H�p  ?Q��5����C'�3                                    Bx�yn�  T          @����aG���  �>�R�ffCG���aG�>�
=�L���(p�C-.                                    Bx�y}z  
�          @�Q��Y�����
�<(���HCL@ �Y��>���S�
�0C1n                                    Bx�y�   �          @�  �HQ��G��G
=�#ffCQL��HQ�=#�
�dz��B�HC3@                                     Bx�y��  
�          @����O\)�˅�Fff�"�\CN#��O\)>8Q��^�R�;�C0��                                    Bx�y�l  
F          @�G��X�ÿ�  �8����COJ=�X�ýu�W��3�C5�                                    Bx�y�            @��\�S33��z��=p��G�CR\�S33�.{�aG��;33C6��                                    Bx�yƸ  "          @�=q�K���Q��C33�\)CSff�K�����g
=�B�\C6��                                    Bx�y�^            @����A��   �Fff�!�
CUxR�A녾8Q��k��J{C7p�                                    Bx�y�  6          @�  �<�Ϳ�(��J=q�&�CU�R�<�ͽ��n{�N\)C6Y�                                    Bx�y�  �          @���7�����S33�133CS�q�7�=����p���RffC2�                                    Bx�zP  �          @�  �<(���\)�Mp��*��CT���<(����
�n{�N��C4Y�                                    Bx�z�  �          @�
=�'����R�\�1�C[��'��L���z=q�`ffC8O\                                    Bx�z�  "          @����Q��'
=�S�
�6�CiE��Q�����p���HCEG�                                    Bx�z-B  �          @���z��#33�Z�H�<=qCi5ÿ�z�   ����CB�=                                    Bx�z;�  
�          @�{���H�(Q��W
=�6CiQ���H�(���
=33CE0�                                    Bx�zJ�  �          @�p����R���^�R�ACe�ÿ��R���
��ffǮC=                                    Bx�zY4  T          @�p��33�	���fff�I�RCbE�33��\)��{�ffC6�                                    Bx�zg�  "          @�\)�ff����i���J�Ca�=�ff�L����\)�~33C5B�                                    Bx�zv�  
�          @�������(��e��BC`���׽���ff�v��C7\                                    Bx�z�&  
�          @����
�H���aG��>z�Ccs3�
�H���R��\)�z\)C<                                      Bx�z��  �          @�Q���H�#�
�_\)�=�Ch�=���H���H������CB                                      Bx�z�r  
�          @�
=���"�\�_\)�>�Ch녿������G��\CA�{                                    Bx�z�  �          @�
=����&ff�]p��<�RCj���녿
=q��G��CC��                                    Bx�z��  �          @�Q�����\���:\)Ca�녾��
�����t(�C<{                                    Bx�z�d  �          @����z���\(��9�Ca0��zᾨ�������r�C<�                                    Bx�z�
  r          @���%����X���3�
C\h��%�u��G��e  C9=q                                    Bx�z�            @��\�#�
�33�X���233C^��#�
�������\�f�C;W
                                    Bx�z�V  T          @���&ff�(��Y���4  C\��&ff�k���G��d�C8�q                                    Bx�{�  �          @��R�,(�����X���9p�CU�{�,(�=��u��[z�C1�                                    Bx�{�  �          @�  �8Q쿮{�aG��?=qCML��8Q�?���n�R�OQ�C):�                                    Bx�{&H  �          @�p��8Q쿱��n�R�Ep�CMǮ�8Q�?�R�{��Tz�C'�                                    Bx�{4�  �          @���:�H�����j�H�EG�CJ^��:�H?B�\�r�\�M�HC%c�                                    Bx�{C�  
�          @����G
=��G��n�R�@�CJ��G
=?:�H�w��I��C&��                                    Bx�{R:  T          @�z��2�\��\)�i���B33CR��2�\>Ǯ�~�R�Z�C,�                                    Bx�{`�  �          @�ff�  �����l(��O��CZ��  >k�����t��C.=q                                    Bx�{o�  �          @��R��\��p��g
=�H=qC\�f��\=L�����
�s�HC2�)                                    Bx�{~,  T          @�  ��ÿ�Q��g
=�F33C[+����=�\)��33�o(�C28R                                    Bx�{��  "          @����=q����l���I��CZ�=q>.{�����oQ�C/��                                    Bx�{�x  "          @�G��p������k��H�HCX���p�>W
=��33�kC/!H                                    Bx�{�  �          @�����
�G��j=q�H�C]���
<����t�C3@                                     Bx�{��  �          @�=q�����
�h���DffC\����ü��
���p��C4h�                                    Bx�{�j  
�          @�33�
=�z��l(��FG�C]O\�
=�#�
��\)�s\)C4E                                    Bx�{�  
�          @��� �׿��H�j�H�DG�CY��� ��=��
����k�C2(�                                    Bx�{�  "          @�p���R���l(��B{C\}q��R�L����  �n��C55�                                    Bx�{�\  
�          @����������i���A�C^�)��þ\)��Q��r�RC7c�                                    Bx�|  �          @���\)����h���D��C`�{�\)�#�
��  �xC8�                                    Bx�|�  T          @��/\)��
=�g
=�<�
CW&f�/\)=�\)���H�a  C2z�                                    Bx�|N  T          @�G��J=q��
=�b�\�2�
CP{�J=q>�  �z�H�L{C/��                                    Bx�|-�  	�          @���AG���=q�hQ��7G�CS.�AG�>#�
����U\)C0�f                                    Bx�|<�  	�          @�=q�=p�����j=q�8�
CT� �=p�=����
�Y
=C1��                                    Bx�|K@  �          @��\�AG������h���7p�CSxR�AG�>�����\�V
=C1+�                                    Bx�|Y�  	�          @����?\)��  �^{�4G�CRW
�?\)>���xQ��Q�\C1(�                                    Bx�|h�  �          @�Q��?\)����c33�4G�CTG��?\)=u�����UQ�C2�)                                    Bx�|w2  
(          @����;��33�dz��3��CV���;���\)����Z33C5h�                                    Bx�|��  �          @��
�9�����j=q�6��CW���9���u���R�]��C5!H                                    Bx�|�~  T          @�(��;��G��l(��833CV�\�;�<#�
��ff�\�\C3�\                                    Bx�|�$  �          @�(��?\)��\�h���4��CVL��?\)�#�
��p��Yp�C4��                                    Bx�|��  �          @�
=�O\)���`  �(=qCU+��O\)�8Q����\�N
=C7�                                    Bx�|�p  �          @����Tz��{�\���#33CU���Tzᾔz����H�K=qC8��                                    Bx�|�  �          @�G��N�R���`���&33CW0��N�R���
��p��PQ�C9�
                                    Bx�|ݼ  �          @����J=q�G��c�
�)�CW���J=q��\)���R�T  C9�                                    Bx�|�b  �          @�  �J�H�  �a��(�RCWY��J�H��z���p��R�\C9(�                                    Bx�|�  �          @�  �Mp��33�\���$�\CW���Mp���p���(��O�
C:�=                                    Bx�}	�  �          @�G��J�H�{�\(��"(�CY�R�J�H����ff�Rp�C=u�                                    Bx�}T  
�          @����G
=�%�X����HC[� �G
=�&ff���R�S��C?ٚ                                    Bx�}&�  �          @��
�Q��!��Z=q�CY�f�Q녿����ff�NG�C>Q�                                    Bx�}5�  �          @�ff�N�R�1��X�����C\�R�N�R�Tz������P�CBc�                                    Bx�}DF  �          @��Q��'��\����RCZ�H�Q녿(�������P
=C?aH                                    Bx�}R�  �          @�{�R�\�#�
�_\)��CY�
�R�\�
=��G��PG�C>
                                    Bx�}a�  �          @����O\)�*�H�Y���C[z��O\)�=p���Q��P  C@��                                    Bx�}p8  �          @��^�R�'��N�R�
=CX�R�^�R�G����H�Cp�C@�=                                    Bx�}~�  �          @�
=�g��%��K��\)CWu��g��B�\�����=��C?޸                                    Bx�}��  �          @��R�g��+��Fff�
33CXxR�g��c�
��  �<
=CA                                    Bx�}�*  �          @�ff�e�$z��J=q���CW�
�e�B�\��  �>
=C@�                                    Bx�}��  �          @�{�e�%��H���Q�CW���e�J=q�~�R�={C@ff                                    Bx�}�v  
�          @�p��`  �"�\�P  ���CW���`  �333��=q�B�HC?Q�                                    Bx�}�  T          @�
=�b�\�.{�J�H��\CY�\�b�\�fff���\�@�CBO\                                    Bx�}��  
F          @�Q��XQ��9���O\)�=qC\���XQ쿃�
��\)�H�\CD�q                                    Bx�}�h  �          @����Q��=p��U�\)C^��Q녿�����H�N\)CE��                                    Bx�}�  T          @����Tz��>�R�P����\C]���Tzῌ�������J�CFG�                                    Bx�~�  T          @���Vff�@���H���\)C]���Vff��Q���{�F��CGz�                                    Bx�~Z  �          @���N{�E��Mp��{C_���N{���H�����L�CH�\                                    Bx�~    �          @�Q��K��J=q�L���
=C`�{�K��������M�
CJ�                                    Bx�~.�  T          @����H���QG��I���  Cb{�H�ÿ�z�����M\)CL.                                    Bx�~=L  "          @��_\)�H���,������C]���_\)��G��w
=�5z�CKp�                                    Bx�~K�  T          @�\)�^�R�N{�.�R��C^�{�^�R�����z�H�733CLQ�                                    Bx�~Z�  �          @�ff�U��L���8��� �C_���U���p������@  CK�                                    Bx�~i>  "          @��R�J=q�O\)�B�\��Ca���J=q������ff�I�CL�)                                    Bx�~w�  
�          @���J=q�XQ��<����Cc  �J=q��\)���FCO33                                    Bx�~��  �          @�
=�N{�Z�H�3�
���Cb�q�N{��(����\�@CP)                                    Bx�~�0  �          @�
=�P  �Y���2�\��\)CbE�P  ���H�����?�CO�                                    Bx�~��  �          @�p��C�
�\(��5��p�Cdc��C�
��p����
�F{CQ}q                                    Bx�~�|  T          @����>{�W��?\)���Cd���>{������
=�Mz�CPG�                                    Bx�~�"  �          @��'��U�XQ��  Cg�)�'������G��c=qCP�                                    Bx�~��  �          @�p��0  �X���L����
Cf�3�0  ���
����Y��CQ                                    Bx�~�n  �          @�{�)���W
=�U��HCg�R�)����Q������a{CPk�                                    Bx�~�  �          @���+��R�\�e� �Cf�H�+���G���ff�gG�CM@                                     Bx�~��  �          @��\�#�
�Q��mp��'�Ch�#�
��Q�����nQ�CM                                    Bx�
`  �          @��H�*=q�S33�h���"�Cg5��*=q��  ��Q��i33CMJ=                                    Bx�  �          @�=q�(���[��`  ��HChff�(�ÿ�
=���eQ�CP�                                     Bx�'�  "          @�G��+��g
=�O\)��Cin�+���(������[�
CT��                                    Bx�6R  �          @��R�%��b�\�P  �(�Ci���%���z���Q��_(�CT��                                    Bx�D�  �          @��
�!G��b�\�I�����Cj�=�!G��ٙ���p��^�CU�R                                    Bx�S�  
�          @��#�
�e�J�H�  Cj��#�
�޸R��
=�](�CV+�                                    Bx�bD  �          @�p��!��g��H����RCk��!녿��
��ff�\��CW8R                                    Bx�p�  �          @����"�\�dz��I����Cj���"�\�޸R���\�HCVu�                                    Bx��  �          @���#�
�b�\�L�����Cj!H�#�
��Q����R�]�CUu�                                    Bx��6  T          @�{�%��fff�I�����Cjff�%���\��ff�[��CVs3                                    Bx���  �          @����$z��g��E��  Cj�\�$z������z��YG�CW:�                                    Bx���  T          @����7
=�qG��&ff��=qCh�
�7
=�������A�CYJ=                                    Bx��(  �          @���6ff�l(��(����z�ChY��6ff������C(�CX@                                     Bx���  �          @����8Q��o\)����Chc��8Q��G��r�\�7�\CZ=q                                    Bx��t  �          @�  �7��p�������Q�Ch�)�7����n{�4�C[                                      Bx��  T          @�G��7��u������=qCi)�7�����p  �4Q�C[                                    Bx���  �          @�\)�8Q��r�\�
=q�Ù�Ch���8Q�����h���0C[�f                                    Bx߀f  �          @�ff�7��p���
�H��33Ch�H�7����hQ��1�C[�
                                    Bx߀  �          @��H�7��x������ȏ\Ci�
�7��p��q��3�C\��                                    Bx߀ �  �          @�z��3�
�w
=��R��33Ci���3�
���|���<�C[�R                                    Bx߀/X  �          @��,���fff�C33��\Ci#��,�Ϳ����H�T(�CVW
                                    Bx߀=�  �          @�
=�5�o\)�333��Q�ChǮ�5�����Gz�CXG�                                    Bx߀L�  �          @��\�0���l(��+���\)Ci.�0��������E�RCY33                                    Bx߀[J  �          @���2�\�dz��1G���p�Ch��2�\�������\�H�CW                                    Bx߀i�  T          @��!G��Z=q�=p���
Ci}q�!G���(���p��WG�CVQ�                                    Bx߀x�  �          @�z��%��W
=�:�H�
z�Chz��%��ٙ����
�Tz�CUaH                                    Bx߀�<  �          @�z��#�
�[��6ff�z�CiB��#�
��ff���\�Q��CW�                                    Bx߀��  
�          @�  �Z=q�Fff�������C^G��Z=q��33�HQ���CQ0�                                    Bx߀��  "          @�(��aG��L(���
=����C^(��aG��   �H���ffCQ�\                                    Bx߀�.  �          @����\���N{��ҏ\C_�\�Ϳ����aG��'��CP(�                                    Bx߀��  "          @�
=�e�Q��   ��\)C^c��e��=q�l(��)��CO                                      Bx߀�z  �          @�ff�^�R�R�\�&ff���
C_xR�^�R��ff�r�\�0=qCOff                                    Bx߀�   �          @��\���W
=�\)�ڣ�C`=q�\�Ϳ�z��n{�-G�CP��                                    Bx߀��  "          @��\(��U��$z���C`!H�\(������qG��0�CP\)                                   Bx߀�l  	          @���XQ��X���!G���  Ca��XQ��
=�p  �/�HCQ��                                   Bx߁            @��O\)�Tz��3�
��Ca�R�O\)�޸R�\)�=G�CPB�                                    Bx߁�  k          @�ff�N{�W
=�5��G�Cb5��N{��G������>��CP�R                                   Bx߁(^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁7              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁E�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁TP              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁b�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁q�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁�B              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁ɀ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߁�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂!d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂0
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂MV              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂j�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂yH              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂�,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂�x              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߂�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃j              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃)              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃7�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃F\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃U              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃c�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃rN              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃�~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߃�$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄0�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄?b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄N              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄kT              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄�*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߄��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅)�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅8h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅G              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅U�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅dZ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅s               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅ي              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅�0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߅��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆"�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆1n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆N�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆]`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆�R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆�D              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆Ґ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߆��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇*t              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇9              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇G�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇Vf              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇e              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇s�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇�J              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇˖              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߇��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈#z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈2               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈Ol              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈l�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈{^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈�P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈Ĝ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈�B              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߈�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉+&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉9�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉Hr              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉W              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉e�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉td              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉�H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx߉�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ$,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ2�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxߊAx              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxߊP              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊmj              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ�\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߊ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ:~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxߋI$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxߋW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋfp              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋu              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ�b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ۠              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߋ��  �          @����%��(��L���(�C_u��%�����s�
�SQ�CNٚ                                    Bxߌ�  
�          @���#33�=q�K��(��C_L��#33���
�q��S�CN�H                                    Bxߌ8  �          @���%��H�O\)�)�
C_  �%���\�u�T=qCN!H                                    Bxߌ$�  
�          @����"�\�=q�P  �+ffC_p��"�\��G��u�V�CNn                                    Bxߌ3�  
�          @���(����H�L���'{C^���(�ÿ�ff�s33�Q{CN+�                                    BxߌB*            @�ff�8���
�H�W��+��CX޸�8�ÿ�G��xQ��N�HCG&f                                    BxߌP�  
�          @��
�=p����H�U�,��CU�
�=p��Q��q��K�
CC�=                                    Bxߌ_v  T          @����2�\��ff�G
=�,��CT���2�\�=p��`���J��CB��                                    Bxߌn  "          @��
�ff��
�2�\�'  Cc�=�ff��=q�W��VffCT^�                                    Bxߌ|�  
�          @���1녿�p��;��+�RCO��1녾��H�P  �C�RC>                                    Bxߌ�h  �          @�{�Z�H����1���C=��Z�H>�{�3�
��C.T{                                    Bxߌ�  T          @�{�^�R���-p����C;xR�^�R>����.{�(�C-^�                                    Bxߌ��  "          @�p��U��#�
�5���C>�\�U�>�=q�8���#�\C/W
                                    Bxߌ�Z  T          @�33�H�ÿ����3�
�!�CF�)�H�þ���@  �.�
C6��                                    Bxߌ�   �          @�
=�P�׿=p��"�\���C@�\�P��=��
�)���=qC2�f                                    BxߌԦ  �          @Y����������5
=Cqff��녿�  �:�H�o�Cc��                                    Bxߌ�L  �          @W
=������{�&ffC�� ��Ϳ���7
=�kG�C{h�                                    Bxߌ��  T          @R�\�k���R����"�RC�\)�k����H�333�i�\C�,�                                    Bxߍ �  �          @c�
���H�.{��R��C��q���H��z��<���a
=C�                                    Bxߍ>  �          @^�R�B�\�!��{���C}O\�B�\�޸R�8Q��bCv^�                                    Bxߍ�  T          @Q녿�ff�
�H�   �G�Co5ÿ�ff���H�$z��R�HCdxR                                    Bxߍ,�  �          @Z=q�xQ��\)���Cx���xQ��p��/\)�XG�Cp��                                    Bxߍ;0  T          @i����R�1G���\��C�����R��Q��AG��`{C|8R                                    BxߍI�  
�          @hQ�:�H�{� ���1  C}��:�H�����H���s(�Cuc�                                    BxߍX|  �          @g�<��
�E�������C�&f<��
����(���>��C�1�                                    Bxߍg"  )          @mp���Q��P  ��\���HC�=q��Q��#�
�*�H�8�C�
=                                    Bxߍu�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߍ�n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxߍ�  =          @g�>L���8���
=q��RC��>L���
=�;��X(�C���                                    Bxߍ��  
Z          @:�H>���,(��fff���
C��>���33���{C�H�                                    Bxߍ�`  u          @8��>�\)�"�\��=q���C�&f>�\)�����2�
C��                                    Bxߍ�  
9          @C�
�L���
=��(��33C�UýL�Ϳ��%��dp�C�\                                    Bxߍͬ  
�          @A녾��ÿ�Q���\�EQ�C�33���ÿ��0��p�C~^�                                    Bxߍ�R  T          @4z�>��
�z����'�RC�W
>��
������k��C�G�                                    Bxߍ��  
�          @B�\?aG���\���
�
=C��f?aG���z��Q��N�HC��)                                    Bxߍ��  
�          @Dz�?\(��z����+\)C�T{?\(���{�&ff�h��C�0�                                    BxߎD  T          @XQ�@�����=u?�ffC��=@�����
=�'33C�Q�                                    Bxߎ�  "          @\(�@,�Ϳ��H?Tz�A_
=C�f@,����>aG�@eC��
                                    Bxߎ%�  �          @Z�H@(Q��G�?E�AP��C�0�@(Q��
=q>\)@ffC�H�                                    Bxߎ46  T          @`  @���#33=�@�C��q@���{�!G��%p�C��                                    BxߎB�  �          @^{@ff�!�>W
=@`��C�p�@ff��R�
=q��HC��R                                    BxߎQ�  �          @0  @����?��\A�ffC�#�@���ff?
=AL(�C��f                                    Bxߎ`(  �          @@  @33��z�?ǮB G�C���@33��ff?�=qA���C�^�                                    Bxߎn�  �          @W�?�Q�&ff@+�BR\)C���?�Q쿳33@�B2�
C�                                    Bxߎ}t  
�          @h��@Q쿗
=@,(�B?��C�}q@Q��z�@\)B
=C��                                    Bxߎ�  �          @u@�R��@+�B1��C�XR@�R��33@�RB(�C�C�                                    Bxߎ��  �          @z�H@\)����@4z�B7�C�q@\)��{@Q�B��C��R                                    Bxߎ�f  "          @tz�@   �k�@$z�B0
=C��@   ��\)@(�B��C�}q                                    Bxߎ�  �          @�  @(���Q�>aG�@@��C�h�@(���Mp��:�H� (�C��)                                    BxߎƲ  �          @�p�@,���`  =�G�?�33C���@,���XQ�fff�<z�C�K�                                    Bxߎ�X  �          @�\)@p��n�R=��
?�=qC���@p��fff�}p��L��C�,�                                    Bxߎ��  �          @�@��xQ���˅C�Q�@��k���p���
=C���                                    Bxߎ�  "          @��
?���}p���
=���C�'�?���k�������HC���                                    BxߏJ  �          @�ff?�(��~{��{��C�0�?�(��mp����H����C��q                                    Bxߏ�  T          @���?�\)��Q�c�
�4��C�xR?�\)�fff���R��p�C��
                                    Bxߏ�  "          @��H?���G������d��C�E?��c33��R��G�C���                                    Bxߏ-<  
{          @���?���u��33��ffC�|)?���Mp��*�H�
=C���                                    Bxߏ;�  
�          @��H@�\��  �n{�;�C���@�\�dz��G���C��)                                    BxߏJ�  �          @�\)?�ff�|(�����aG�C�C�?�ff�^{�	�����HC���                                    BxߏY.  
�          @�33?޸R���
��=q�X��C�o\?޸R�h���(���(�C��)                                    Bxߏg�  �          @���?�\�����u�C�
C���?�\�g���
����C�
=                                    Bxߏvz  �          @��?�����녿�\)�b�\C��?����e��{��G�C�w
                                    Bxߏ�   "          @��\?��H��G���G���
C�p�?��H�a������C��                                    Bxߏ��  �          @��\?�(���G������\)C��=?�(��`������C��                                    Bxߏ�l  �          @��?��
���ÿ������C��{?��
�Z�H�*=q�(�C�Ff                                    Bxߏ�  �          @�33?޸R���E����C�G�?޸R�s33��33��C�G�                                    Bxߏ��  T          @��
?��R���
�z���{C���?��R�s33�ٙ���33C��{                                    Bxߏ�^  �          @��?�\)���Ϳ!G���(�C�q?�\)�s�
��G���  C�                                    Bxߏ�  �          @�33?����;�
=���C�p�?��w��Ǯ���RC�8R                                    Bxߏ�  "          @��R?�p������G����HC�q�?�p��z�H��  ���
C���                                    Bxߏ�P  "          @�
=?����R=���?��C�Q�?����\����]p�C���                                    Bxߐ�  T          @�=q?\(�����=��
?�=qC��{?\(���Q쿈���ip�C�f                                    Bxߐ�  
�          @��R?����H��?�G�A�{C�b�?����`��?Q�AFffC�P�                                    Bxߐ&B  T          @���@j=q�z�@-p�B��C�t{@j=q���@(�B ��C�+�                                    Bxߐ4�  "          @�{@L(����@7�BG�C�R@L(��5@�A�{C�%                                    BxߐC�  �          @�\)@�{��
>�@��HC���@�{�
=������C�K�                                    BxߐR4  
�          @��H@��R�zᾅ��EC���@��R�
=q�aG��(��C�q�                                    Bxߐ`�  
�          @��\@U��33�N{�)p�C���@U�k��Z=q�6(�C��                                    Bxߐo�  "          @�Q�?�>�����H�)AK33?�?�������qp�B(��                                    Bxߐ~&  T          @�(�?�  ?333��33��B
=?�  ?����\)�x��Bv=q                                    Bxߐ��  T          @��@p������  �f��C�33@p�?�\��Q��g�A<                                      Bxߐ�r  �          @�33@
=����z=q�i�
C�f@
=?(���xQ��f��Ay��                                    Bxߐ�  �          @���@ff��
=�~�R�v��C�U�@ff?+��|(��s  A�33                                    Bxߐ��  
�          @���@z���y���m=qC���@z�?��\�qG��`ffA�Q�                                    Bxߐ�d  �          @�p�@ff�\)�n{�f��C�K�@ff?^�R�g��]33A��                                    Bxߐ�
  �          @�33@�R���
�dz��\�HC���@�R?s33�\(��Qz�A���                                    Bxߐ�  "          @��R@4z὏\)�\(��J�\C�C�@4z�?Y���U�B{A��R                                    Bxߐ�V  
�          @w�?�p��@  �QG��p  C�Q�?�p�>#�
�Vff�z=q@��                                    Bxߑ�  �          @e�?��}p��E��r�RC��3?����N�RffC�j=                                    Bxߑ�  �          @p  ?��R��z��J=q�l\)C��?��R��=q�VffB�C���                                    BxߑH  
�          @`��?���(��H���~33C���?��z��Z�HG�C�Ф                                    Bxߑ-�  "          @aG�>��R��(��K�B�C���>��R�z��\���=C�=q                                    Bxߑ<�  �          @dz�>�(����
�K��}�C��>�(��!G��^{=qC�W
                                    BxߑK:  
�          @fff����(��>{�b(�C�~�����{�X����C��                                    BxߑY�  
�          @vff����'��3�
�;z�C�&f������
�Y���xp�C��3                                    Bxߑh�  
�          @�  ����%��AG��D�\C��׾�녿�Q��fff\)C�9�                                    Bxߑw,  T          @�33�.{�/\)�A��?�C�E�.{���i���|�\C�o\                                    Bxߑ��  �          @����333�'��@  �?�HC
�333��p��e�y��Cx�                                    Bxߑ�x  
�          @��׿h���.�R�3�
�1ffC{�\�h�ÿ���[��j{CtT{                                    Bxߑ�  T          @�=q�G��:�H�-p��'�RC~�G��ff�XQ��aCy��                                    Bxߑ��  
�          @�z�
=�Dz��0���%�\C��\�
=�\)�^{�`�HC@                                     Bxߑ�j  
�          @}p����H�P���
=q��C��H���H�%��<(��@�C���                                    Bxߑ�  
�          @j=q��p��;��
=�G�C�j=��p����333�J=qC�ff                                    Bxߑݶ  �          @Y��?s33�H�þ�33����C�j=?s33�<(���z����
C��3                                    Bxߑ�\  
�          @j�H>L���B�\��������C��f>L���%��Q���C�>�                                    Bxߑ�  T          @k�?\)�Z=q������(�C��\?\)�@�׿�� p�C�J=                                    Bxߒ	�  
�          @�(���Q��:=q�9���3ffC����Q���
�c�
�o�C���                                    BxߒN  
�          @��;�{�z��Z=q�]�C�� ��{��{�y��u�C�
=                                    Bxߒ&�  "          @���{�*�H�L(��G=qC�W
��{��G��q�8RC�xR                                    Bxߒ5�  T          @�p��xQ��8���5�+�C{\)�xQ���
�_\)�cG�Ct�                                    BxߒD@  �          @�ff��ff�
=�S�
�P
=Cv{��ff��
=�s�

=Ci                                    BxߒR�  T          @�  ?p���������
=C�Q�?p���l�������C��                                    Bxߒa�  �          @�
=?�{��zῢ�\��C���?�{�k���\��Q�C�g�                                    Bxߒp2  �          @�ff?W
=�|(�������C�  ?W
=�Vff�/\)��HC�                                    Bxߒ~�  "          @��
?z�H�e�\)���RC���?z�H�8���E�5(�C�e                                    Bxߒ�~  
�          @��
?E��=p��E�5{C�P�?E��z��o\)�m��C�9�                                    Bxߒ�$  T          @�?E��HQ��A��,��C���?E��  �n�R�e��C�s3                                    Bxߒ��  T          @�z�?z�H�S�
�.�R�G�C�<)?z�H�   �^�R�Q=qC��=                                    Bxߒ�p            @��R?����S�
�����HC�  ?����'��A��9�C�S3                                    Bxߒ�  �          @��?�(��aG���p����C���?�(��:=q�3�
�'  C�b�                                    Bxߒּ  "          @�G�?E��]p���z����C�@ ?E��7��.{�*{C�~�                                    Bxߒ�b  �          @��?�{�^{��p�����C��?�{�>�R��
���C�<)                                    Bxߒ�  
�          @�=q?Tz��6ff�I���:�\C�#�?Tz��(��qG��q�HC�~�                                    Bxߓ�  T          @�
=>�ff�%�Q��MffC��R>�ff���u�L�C��3                                    BxߓT  	�          @���?z��U�(Q��33C��?z��$z��XQ��P�C�`                                     Bxߓ�  T          @�Q�>��e������C��R>��:�H�A��6��C���                                    Bxߓ.�  �          @��R?.{�j=q����{C�=q?.{�C�
�1��&�C�<)                                    Bxߓ=F  T          @{�?���Mp����
�v�RC�1�?���6ff��ff��=qC��R                                    BxߓK�  
�          @z�H?����H�ÿ\����C��H?����)������Q�C��f                                    BxߓZ�  T          @l��?�(��\(�=�?�{C��f?�(��W
=�@  �<��C��)                                    Bxߓi8  
(          @n{@��G��aG��l��C�|)@��	���G��W33C�L�                                    Bxߓw�  
�          @n�R?�  �Fff��=q�ʏ\C���?�  �&ff�33�=qC��3                                    Bxߓ��  
�          @g
=@�
�33������ffC���@�
��33���C���                                    Bxߓ�*  T          @c33@S33��Ϳk��u�C�L�@S33�������\��G�C�e                                    Bxߓ��  �          @c33@X�þ��H�O\)�T��C�� @X�þ�\)�h���n�RC��                                    Bxߓ�v  �          @c�
@?\)��  ��33���C���@?\)�!G������݅C�3                                    Bxߓ�  "          @u@:=q>��G��
=@$z�@:=q?!G���
=���AB�\                                    Bxߓ��  T          @z�H@X�ÿ�zῳ33���RC���@X�ÿE������=qC���                                    Bxߓ�h  "          @s�
?�
=�?\)��33����C�h�?�
=�'
=������G�C�5�                                    Bxߓ�  �          @z�H?��
�O\)�����(�C�n?��
�4z��33����C�*=                                    Bxߓ��  �          @��@
=q�E���R��Q�C���@
=q�(Q������C���                                    Bxߔ
Z  
�          @�p�@33�I��������
=C�
@33�,��������C�AH                                    Bxߔ   �          @�\)@%�8Q��z����C��{@%����z����C���                                    Bxߔ'�  
�          @���@1��7���G�����C��@1��=q�
�H��  C��f                                    Bxߔ6L  
�          @�Q�@��A녿���=qC�Y�@��!��
=��HC��=                                    BxߔD�  �          @���@�\�Y������x��C��q@�\�AG���Q��أ�C���                                    BxߔS�  "          @��R@ff�a녿aG��@��C�b�@ff�Mp���(����\C��)                                    Bxߔb>  	�          @�
=@{�_\)�J=q�,��C�8R@{�L(���\)����C�ff                                    Bxߔp�  �          @�ff@z��P�׿�  ��z�C���@z��6ff�   ��ffC���                                    Bxߔ�  T          @�@33�Z�H��p�����C�}q@33�AG������C��                                    Bxߔ�0  T          @�{?����[���33��(�C���?����?\)�(���z�C��=                                    Bxߔ��  T          @�?�ff�]p���  ���HC��H?�ff�?\)�33��C���                                    Bxߔ�|  
�          @�p�@��U����\���C�f@��:�H��\����C��                                     Bxߔ�"  �          @�
=?�=q�a녿�{��33C��3?�=q�Fff����\)C�K�                                    Bxߔ��  T          @�Q�?��g����R��  C�c�?��I���z��33C��
                                    Bxߔ�n  
�          @�Q�?�Q��`�׿��H���
C���?�Q��?\)� ����\C��{                                    Bxߔ�  
�          @�  ?�\)�c�
�У�����C�9�?�\)�Dz��(���\C��                                    Bxߔ��  
�          @���?��dz΅{��z�C�&f?��HQ�����G�C��                                     Bxߕ`  T          @�  ?�(��^�R���H��ffC���?�(��AG��  ��C���                                    Bxߕ  
�          @��?޸R�g
=��z����C�� ?޸R�J=q�\)����C�j=                                    Bxߕ �  �          @���?ٙ��g��\��
=C��\?ٙ��H���ff�33C�+�                                    Bxߕ/R  T          @���?˅�c33�޸R����C�{?˅�A��"�\�p�C��q                                    Bxߕ=�  �          @��?Ǯ�`  ��(���\)C��?Ǯ�:�H�0  ��C��                                    BxߕL�  "          @�G�?�Q��^�R��\�㙚C�E?�Q��8���3�
�$33C�Ff                                    Bxߕ[D  T          @���?���R�\�"�\�Q�C��?���%�P  �EQ�C�R                                    Bxߕi�  T          @�Q�?�z��U��Q��(�C��H?�z��*�H�G
=�;��C��=                                    Bxߕx�  T          @�\)?�G��X��� ���㙚C��?�G��3�
�0���#ffC�                                      Bxߕ�6  
�          @�z�?�Q��Y����{��  C�s3?�Q��7
=�'��
=C�U�                                    Bxߕ��  T          @{�@0  �*=q?^�RALz�C���@0  �2�\>W
=@C�
C�C�                                    Bxߕ��  
�          @|(�@.{�1�?�A�\C�'�@.{�5���Ϳ�  C��q                                    Bxߕ�(  �          @���@"�\�C33=�Q�?��RC��@"�\�?\)�#�
�z�C�7
                                    Bxߕ��  �          @�  @z��L�Ϳ���\)C�n@z��6ff������Q�C���                                    Bxߕ�t  T          @|(�?�
=�Tz��\��G�C��?�
=�G
=���
��Q�C��3                                    Bxߕ�  
�          @�G�?�
=�aG����
�k�C�U�?�
=�Z�H�Y���C\)C���                                    Bxߕ��  
�          @�G�@   �^�R��
=���C��\@   �Q녿�p����C��=                                    Bxߕ�f  
�          @���@��U�8Q��%��C��
@��Dz῾�R���\C��                                    Bxߖ  "          @��?����QG�������C�k�?����7��G���C�R                                    Bxߖ�  
�          @�p�?��R�B�\�����C��?��R�Q��E��>ffC��)                                    Bxߖ(X  D          @��H?�33�J=q���R��p�C��?�33�'
=�+��#��C�*=                                    Bxߖ6�  
j          @�ff@��P�׿�����=qC�~�@��7
=�33��C�K�                                    BxߖE�  "          @�{@(��Mp���\)�xQ�C�� @(��7
=������C�AH                                    BxߖTJ  �          @�G�@-p��L(��n{�H��C�0�@-p��8Q��33��p�C���                                    Bxߖb�  
�          @��@5�K��aG��;
=C�� @5�8Q��������C�C�                                    Bxߖq�  
�          @�33@1G��DzῙ�����
C��@1G��,�Ϳ����(�C���                                    Bxߖ�<  T          @���@0  �8Q��\)��p�C��
@0  �
=�\)�
�C���                                    Bxߖ��  T          @�z�@(Q��7��G��ۅC�C�@(Q��z��(Q���RC�P�                                    Bxߖ��  �          @�G�@/\)�333�޸R����C�1�@/\)�z��ff��C��f                                    Bxߖ�.  �          @�z�@6ff�4z��p����\C��@6ff����C�N                                    Bxߖ��  
�          @�@4z��;����H��z�C��@4z�����ff� z�C�}q                                    Bxߖ�z  "          @�p�@333�8�ÿ���p�C��@333������C���                                    Bxߖ�   
�          @�=q@(���Fff�����C�7
@(���,(��
=��C�C�                                    Bxߖ��  �          @��\@1G��AG��������\C�@ @1G��'���\��p�C�K�                                    Bxߖ�l  
�          @��H@%�N�R���
����C�Y�@%�6ff�   �ۅC�%                                    Bxߗ  
�          @��H@*�H�N{����i�C���@*�H�8Q����C�q�                                    Bxߗ�  "          @�Q�@(Q��E��Q���33C�9�@(Q��.�R�������C���                                    Bxߗ!^  �          @�=q@*=q�E������33C�e@*=q�+��33��  C�c�                                    Bxߗ0  �          @���@)���@�׿�(����C��@)���%�Q���\)C��R                                    Bxߗ>�  "          @��@$z��8Q������C�޸@$z��
=�!���C���                                   BxߗMP  
Z          @���@!G��4z����ᙚC�� @!G����(Q���
C���                                    Bxߗ[�  T          @���@p��1�������C�@ @p��
=q�>{�/\)C��H                                    Bxߗj�  
�          @��H@Fff�#33����~ffC�N@Fff�p��ٙ���Q�C�<)                                    BxߗyB  
�          @�=q@Mp��*=q���
�^=qC�*=@Mp��ff��\)����C��                                    Bxߗ��  
�          @�G�@8���1녿�p���  C�\@8����������C�\)                                    Bxߗ��  
�          @��H@C33�/\)����\)C��@C33���G��ݙ�C�G�                                    Bxߗ�4  �          @�33@Fff�,�Ϳ�{��=qC�� @Fff��
��Q���33C��f                                    Bxߗ��  
�          @��\@J�H�(�ÿ�=q��33C��@J�H��׿�33��ffC�<)                                    Bxߗ  2          @���@H���,(���z��z{C��{@H���ff�޸R��C��{                                    Bxߗ�&  
�          @�G�@G��#�
���H���C�K�@G��
=q� ����\)C���                                    Bxߗ��  T          @��@1G��DzῚ�H���HC�  @1G��-p�����ϮC���                                    Bxߗ�r  
Z          @���@/\)�>�R��\)��C�O\@/\)�%��G���\)C�W
                                    Bxߗ�  T          @���@+��@�׿����RC��{@+��'
=�z���C���                                    Bxߘ�  
Z          @���@AG��/\)������
C��@AG��
=��Q���Q�C���                                    Bxߘd  T          @�=q@Q�� �׿�p����HC�P�@Q��
=q��\��Q�C�U�                                    Bxߘ)
  
(          @��\@\(��G���ff���C�N@\(���z���
�\C�xR                                    Bxߘ7�  �          @�33@^{��R�������
C��)@^{��{�����ɮC��f                                    BxߘFV  
�          @��H@fff�Q쿎{�n=qC��{@fff���ÿ�����{C���                                    BxߘT�  
�          @��\@aG��޸R��
=��\)C��R@aG��������p�C���                                    Bxߘc�  "          @��H@g��ff��\)�o�C��3@g�����������
C��)                                    BxߘrH  T          @��@fff����������=qC��H@fff���ÿ�{�ʸRC�5�                                    Bxߘ��  T          @��H@aG��G����R��\)C�)@aG���\)��z����
C��H                                    Bxߘ��  T          @��@`�׿�Q��z����C���@`�׿�G���
��Q�C�c�                                    Bxߘ�:  
�          @�33@l�Ϳ�p���G���z�C�w
@l�Ϳ����{����C�
=                                    Bxߘ��  �          @��H@n{��33��G�����C��@n{��G�����\)C��f                                    Bxߘ��  
(          @��H@o\)��ff�������C��)@o\)��33������
C�o\                                    Bxߘ�,  �          @�33@tzῠ  ����ffC��{@tz�W
=��z���(�C��{                                    Bxߘ��  
�          @��H@s33��=q��ff��C�{@s33�#�
���R��
=C�5�                                    Bxߘ�x  �          @���@Q녿��H��33��p�C���@Q녿�p��33�{C���                                    Bxߘ�  �          @���@P  ���z���C��
@P  ��33����
�RC�XR                                    Bxߙ�  T          @���@L(���Q�����
=C�S3@L(���z�� �����C��                                    Bxߙj  �          @��@P  ��z��33��  C���@P  ��
=��\�  C�"�                                    Bxߙ"  �          @���@c33���Ϳ�z�����C�޸@c33��
=��p���G�C���                                    Bxߙ0�  
�          @��@\�Ϳ��
��{�̏\C�Y�@\�Ϳ���p����C��{                                    Bxߙ?\  "          @��@G���p��p���ffC��=@G���
=�&ff�\)C��                                    BxߙN  
�          @�33@L������ ������C�
@L�ͿУ�����	
=C�}q                                    Bxߙ\�  
}          @��\@:=q���
=��\C�h�@:=q��=q�%���C��f                                    BxߙkN  �          @�G�@>{�(���z���C�Ff@>{������H�	
=C�`                                     Bxߙy�  �          @�=q@W���\)��
=��Q�C�|)@W�����33� ��C��3                                    Bxߙ��  
�          @��@\(��   ���
���HC��@\(�����(�����C��                                    Bxߙ�@  
�          @��\@a녿�\)����p�C��@a녿�Q��33���C��f                                    Bxߙ��  
Z          @���@W
=���R���
��ffC���@W
=���
�(����C��R                                    Bxߙ��  �          @��@P�������ʏ\C�|)@P�׿�33�G�����C���                                    Bxߙ�2  �          @�33@O\)�{��{�ʸRC��3@O\)�޸R��
� ��C��                                    Bxߙ��  T          @�G�@I���ff� ����C�/\@I���˅���
G�C��H                                    Bxߙ�~  �          @��
@a��z�\����C��f@a녿�
=������(�C�H�                                    Bxߙ�$  T          @��H@c33����\)����C��)@c33���H��ff��(�C�!H                                    Bxߙ��  �          @��
@`  �   ��z����C�#�@`  �����z���RC���                                    Bxߚp  "          @�(�@`  �
�H��p����
C��@`  ���
��
=��33C��=                                    Bxߚ  
�          @�(�@\����R���R���
C��3@\�Ϳ����H���HC��                                    Bxߚ)�  "          @�(�@P  ��\)�
�H��z�C�@P  ��=q�!��Q�C�޸                                    Bxߚ8b  "          @��@Tz��\)�����HC�P�@Tz΅{����G�C���                                    BxߚG  �          @�z�@Y���z���
����C��\@Y����� ����(�C�B�                                    BxߚU�  �          @��H@]p��  ������33C�}q@]p���׿�������C��{                                    BxߚdT  
�          @�=q@Y���\)��
=����C�J=@Y����{��33���HC���                                    Bxߚr�  �          @��@W���R��{���
C�G�@W����z���C���                                    Bxߚ��  �          @�=q@Vff�
=��{��ffC�c�@Vff�   ������p�C��R                                    Bxߚ�F  �          @��@Q��"�\��z��y�C�q@Q��p��ٙ���Q�C��q                                    Bxߚ��  �          @��H@Mp��*=q��
=�}�C�4{@Mp��z��  ���\C��                                    Bxߚ��  "          @��@N{�,(��xQ��P��C�@N{�=q��ff��G�C���                                    Bxߚ�8  �          @���@Mp��,�Ϳ^�R�:�HC��
@Mp����������RC�h�                                    Bxߚ��  
�          @�  @QG��#�
�Y���8��C���@QG��33��z����HC�j=                                    Bxߚل  
Z          @�ff@XQ��G���  �]G�C�\@XQ���R���R��Q�C���                                    Bxߚ�*  T          @�
=@U�
=��  �\(�C�g�@U�z��G���C�)                                    Bxߚ��  �          @�
=@Vff��H�O\)�0��C��@Vff�
�H��=q��p�C��                                    Bxߛv  �          @�{@S�
�p��:�H�   C���@S�
��R��G����\C��q                                    Bxߛ  T          @�\)@XQ��(��(���\)C��@XQ���R��Q����C�G�                                    Bxߛ"�  
�          @�Q�@Tz��\)�n{�Ip�C��=@Tz��{���H��G�C�)                                    Bxߛ1h  
�          @�
=@J�H�%�����c�C�u�@J�H�녿˅��G�C�+�                                    Bxߛ@  T          @�\)@Fff�'
=��Q���
=C���@Fff�녿�  ���
C��R                                    BxߛN�  T          @��@N{�"�\���
�`��C�ٚ@N{�\)�������RC���                                    Bxߛ]Z  �          @�ff@?\)�0  ��G��`(�C���@?\)��Ϳ�{����C�\)                                    Bxߛl   T          @�  @Dz��'����
����C�@Dz��G�����  C���                                    Bxߛz�  "          @�\)@J�H�#�
�^�R�@��C���@J�H�33��
=��(�C��                                    Bxߛ�L  T          @��@_\)�   ��{��\)C�33@_\)�
=�c�
�?33C���                                    Bxߛ��  �          @���@`  ��������C��{@`  ��+���C�{                                    Bxߛ��  �          @�ff@Vff�!G��\)��Q�C��\@Vff��H�5�33C�R                                    Bxߛ�>  "          @�p�@S33� �׾\����C�` @S33�
=�p���P��C�9�                                    Bxߛ��  �          @��\@AG��Q쿚�H��C�޸@AG��33��(��ɮC��                                    BxߛҊ  �          @~{@8Q쿨��@�B��C��
@8Q����@ ��A��C��                                     Bxߛ�0  �          @x��@;����H?��HA���C�� @;���\)?�=qA�G�C��H                                    Bxߛ��  
�          @~{@U���?�Q�Aə�C��@U��Q�?���A�=qC��                                    Bxߛ�|  
�          @\)@*=q�!G�@7
=B933C�L�@*=q��G�@(��B'��C�H�                                    Bxߜ"  T          @�  @0  ���@)��B'G�C�*=@0  ��z�@�B�HC�j=                                    Bxߜ�  �          @\)@:�H��H?&ffA�C�,�@:�H�   =��
?��\C���                                    Bxߜ*n  �          @�  @2�\�333>��@	��C�xR@2�\�0�׾��H��33C���                                    Bxߜ9  T          @�@0  �C33�#�
�{C�  @0  �5��������C�R                                    BxߜG�  w          @�ff@:�H�=p��
=q����C�O\@:�H�0�׿�������C�O\                                    BxߜV`  1          @�p�@:�H�:�H���H�أ�C���@:�H�.�R����~ffC�w
                                    Bxߜe  
(          @�ff@Fff�2�\=L��?333C��R@Fff�.�R�z�� ��C�J=                                    Bxߜs�  
�          @��@5��@  >�G�@��C��@5��AG���  �`  C��{                                    Bxߜ�R  �          @�p�@-p��H��>���@�Q�C�c�@-p��HQ�Ǯ��C�k�                                    Bxߜ��  "          @��@*�H�C33?Tz�A8��C��R@*�H�J=q=�G�?�=qC��                                    Bxߜ��  T          @���@,���E>\)?�
=C��{@,���B�\����C��\                                    Bxߜ�D  T          @��@��^�R=�?�Q�C�3@��Z=q�+���\C�P�                                    Bxߜ��  
�          @�p�@&ff�O\)���R���RC�\)@&ff�E����
�e�C��                                    Bxߜː  �          @��@z��Z�H��Q�����C�\@z��P  �����}G�C��                                     Bxߜ�6  
�          @�
=@1G��I����\)�u�C��=@1G��@  �}p��Y�C�XR                                    Bxߜ��  T          @�  @{�[��������\C��)@{�QG������k
=C��H                                    Bxߜ��  "          @�
=@"�\�U��33���C���@"�\�K������rffC�K�                                    Bxߝ(  T          @�\)@'��P  �����C�z�@'��AG���=q����C�z�                                    Bxߝ�  T          @��R@$z��Q녿����RC��@$z��Dzΰ�
���C��
                                    Bxߝ#t  "          @��@Dz��4zᾔz��}p�C���@Dz��+��k��L(�C�xR                                    Bxߝ2  �          @�  @Y���{������\)C���@Y�����aG��AC���                                    Bxߝ@�  �          @�G�@6ff�Mp�=#�
?�C���@6ff�HQ�0�����C�(�                                    BxߝOf  �          @���@.{�Tz�>L��@-p�C���@.{�Q녿\)��  C��R                                    Bxߝ^  �          @�G�@8Q��H��>��@�\)C�C�@8Q��I�����
����C�7
                                    Bxߝl�  "          @��H@*=q�N{��33�v�HC��f@*=q�7���{��33C�ff                                    Bxߝ{X  �          @��@+��S33�����b{C���@+��>{��ff����C��                                    Bxߝ��  
�          @��@'��Dz��Q����HC�B�@'��   �1��  C�0�                                    Bxߝ��  
i          @��\@0  �5�33���C��@0  �\)�8����C�k�                                    Bxߝ�J  �          @�z�@%�\)�;���RC�3@%��  �Z�H�>=qC��R                                    Bxߝ��  
�          @�=q@�\��G��l���[
=C��@�\�\�x���l��C�G�                                    BxߝĖ  "          @�=q@�
�k��~{�q��C���@�
<��
��=q�|>�                                    Bxߝ�<  �          @��@z῁G��vff�l=qC��
@z�����~�R�y�
C��{                                    Bxߝ��  �          @�?�{��=q�vff�s  C���?�{�.{�\)��C�h�                                    Bxߝ��  	`          @���?�\)�E���33�C�h�?�\)>L����p�8RA�                                    Bxߝ�.  T          @�p�?���}p����HW
C�g�?�����
���R�=C���                                    Bxߞ�  �          @��H@33��ff�Tz��J=qC�XR@33�}p��i���h
=C��                                    Bxߞz  
�          @�=q@"�\��Q��8Q��(G�C�N@"�\��(��P  �DffC�"�                                    Bxߞ+   
~          @�@�׿�  �QG��C��C��@�׿s33�e�^�HC���                                    Bxߞ9�  
�          @��@=q�;�����\)C���@=q�33�AG��(��C�1�                                    BxߞHl  
�          @�ff@{�E���\��ffC�g�@{�!G��,(��=qC�9�                                    BxߞW  
�          @�p�@33�Mp���(���ffC��
@33�*�H�*=q�G�C�k�                                    Bxߞe�  
�          @���@
=�Q녿�  ���C��@
=�1��p��  C�,�                                    Bxߞt^  �          @�G�?�p��0  �%��RC���?�p��z��J=q�?
=C��q                                    Bxߞ�  x          @���@Q��1G������ffC�]q@Q��
�H�5�%C���                                    Bxߞ��  b          @��H@z��{�,(���\C��{@z���
�L(��=C�9�                                    Bxߞ�P  �          @���@��-p��   ��
C��H@���
�C�
�9G�C���                                    Bxߞ��  �          @�Q�@hQ쿳33?���A�z�C�t{@hQ��p�?��HA��
C�Ff                                    Bxߞ��  T          @���@\�Ϳ�
=?��A�C���@\�Ϳ���?h��AO
=C�B�                                    Bxߞ�B  
�          @��H@��,(�?޸RA�\)C�|)@��AG�?��A\)C��H                                    Bxߞ��  	�          @�ff@Vff����?�  A�
=C��@Vff�
�H?��
A�C���                                    Bxߞ�  0          @�ff@Tz´p�@A�G�C��q@Tz��
=?�Q�A��C��                                    Bxߞ�4  
�          @�\)@W
=��\)?��A�
=C�t{@W
=��?Q�A;�C��                                     Bxߟ�  �          @��\?5�$z��C�
�CC���?5���
�e��w
=C��                                    Bxߟ�  T          @�\)?B�\�*�H�L���Dz�C��=?B�\���n�R�w�RC�%                                    Bxߟ$&  "          @���@$z��-p�?\A��C��{@$z��?\)?W
=AA�C�N                                    Bxߟ2�  �          @�33?�ff�G
=�����HC�G�?�ff�"�\�0  �*�C��                                     BxߟAr  �          @��\@p��5�?��A�Q�C��{@p��L��?�A���C�L�                                    BxߟP  "          @�G�?�  �e��#�
�#�
C�f?�  �^{�aG��L��C�aH                                    Bxߟ^�  
�          @��?�(��l�ͽ�\)�p��C�t{?�(��e��n{�S�
C�Ф                                    Bxߟmd  �          @�z�?\�n�R��R�Q�C��?\�^�R��p���z�C��f                                    Bxߟ|
  
�          @���?�G��xQ�Ǯ��  C��)?�G��k������\)C�q�                                    Bxߟ��  �          @��?��aG�?   @�{C��H?��b�\��{��{C���                                    Bxߟ�V  T          @��?�
=�dz�?   @�  C�33?�
=�e���Q���Q�C�%                                    Bxߟ��  �          @��?���n�R�^�R�B�\C�9�?���Z�H��p��ř�C�"�                                    Bxߟ��  
�          @��R?����Q��Q��	��C�z�?����(Q��E�=
=C���                                    Bxߟ�H  T          @�Q�?Y���XQ�����C��?Y���-p��J=q�@=qC��f                                    Bxߟ��  T          @�ff?L���\���{� p�C���?L���4z��>{�5�C��{                                   Bxߟ�  "          @��R?n{�E�&ff��C�aH?n{����P  �O
=C��f                                   Bxߟ�:  �          @���?z�H�1��E��933C��3?z�H��(��j=q�k�C�8R                                    Bxߟ��  T          @��\?E���`���[�\C�)?E���Q��~{(�C��                                    Bxߠ�  
Z          @�=q?W
=�U��33��{C�
=?W
=�0  �1��0�C�s3                                    Bxߠ,  �          @�  ?�{�c�
���H�݅C���?�{�?\)�0  �"�C�0�                                    Bxߠ+�  �          @�
=?�p��R�\�������C�"�?�p��,(��7��,33C�w
                                    Bxߠ:x  �          @�(�?����Mp���
=��C���?����1��	���
=C�8R                                    BxߠI  �          @�?:�H��z�?��A�33C��q?:�H���H>��R@���C�                                    BxߠW�  �          @���?��R�qG�?\A�C��?��R����?�@���C���                                    Bxߠfj  �          @�p�?���Tz�@�A���C���?���n�R?�G�A�p�C���                                    Bxߠu  �          @tz�?�33�5@{B�C��?�33�R�\?�p�A�  C���                                    Bxߠ��  �          @n�R?���H@Q�B!Q�C�+�?��:�H?�p�A�z�C��\                                    Bxߠ�\  �          @hQ�?c�
��p�@AG�Bd�
C���?c�
�=q@!G�B1��C�"�                                    Bxߠ�  T          @p  ?�=q� ��@\)B)��C�'�?�=q�#33?�
=A��RC���                                    Bxߠ��  �          @z�H@��\@&ffB&C�Ф@�'
=@�A�z�C�U�                                    Bxߠ�N  �          @���@��  @=qB�\C��3@��0��?��A�z�C��                                    Bxߠ��  �          @��H?����,��@�
B33C��R?����J�H?˅A�G�C��                                    Bxߠۚ  �          @\)?��
�5@	��BC�
=?��
�QG�?�33A��RC�B�                                    Bxߠ�@  �          @~�R?�\)�4z�@ffA��\C��?�\)�O\)?���A�
=C���                                    Bxߠ��  �          @��
?��R�*=q@��B(�C�c�?��R�J�H?�
=A���C��                                    Bxߡ�  �          @�=q@
=�!G�@
=B��C��
@
=�AG�?�
=A�(�C�y�                                    Bxߡ2  �          @��
@����@�B��C��q@���7�?�(�A�{C��q                                    Bxߡ$�  �          @�(�@�� ��@ffBG�C�)@��@  ?�A���C���                                    Bxߡ3~  �          @��
@33�p�@ffBz�C���@33�=p�?�
=A���C��=                                    BxߡB$  �          @��
@5��?�Q�A�ffC���@5���?�
=A�(�C���                                    BxߡP�  T          @�@g
=���?�33A�ffC�(�@g
=��33?��
A���C��\                                    Bxߡ_p  
�          @�(�@j�H��33?��HA�(�C�W
@j�H��(�?��A�{C�)                                    Bxߡn  
�          @�z�@e���  ?��\A��HC���@e���\?\(�AC33C��R                                    Bxߡ|�  �          @��@n{��33?���A~�RC��=@n{���?B�\A(z�C�q                                    Bxߡ�b  �          @��
@k��У�?Q�A7
=C��@k����
>Ǯ@�ffC��                                    Bxߡ�  �          @�p�@z=q��\)?c�
AD��C��@z=q��ff?z�A ��C�Ǯ                                    Bxߡ��  T          @��
@x�ÿ���?B�\A*ffC�J=@x�ÿ�p�>��@��HC�@                                     Bxߡ�T  �          @���@w����?�@��RC���@w���>k�@L(�C���                                    Bxߡ��  �          @��@{���(�?
=Ap�C�h�@{�����>�=q@k�C��{                                    BxߡԠ  �          @��@r�\��(�?E�A,  C�n@r�\��{>Ǯ@�33C�|)                                    Bxߡ�F  �          @�{@|(����>\@��C���@|(����<�>�C�N                                    Bxߡ��  �          @�ff@\)���\=�G�?��C�0�@\)��G��W
=�3�
C�C�                                    Bxߢ �  T          @�
=@�Q쿢�\>W
=@7
=C�8R@�Q쿣�
��G��\C�"�                                    Bxߢ8  
�          @��R@vff��
=��=q�l(�C�8R@vff�Ǯ�.{���C���                                    Bxߢ�  �          @�
=@r�\���
����Q�C�q�@r�\���Ϳfff�FffC���                                    Bxߢ,�  �          @�
=@n�R���Ϳz���{C��R@n�R��녿��\�aG�C��                                    Bxߢ;*  �          @�p�@e��33?��Ap(�C��@e��{?(�A
{C�G�                                    BxߢI�  �          @�ff@X�ÿ�33?���A��
C�  @X����
?���A�{C�b�                                    BxߢXv  �          @�@\(���  ?ǮA��C��=@\(���?���An=qC�j=                                    Bxߢg  �          @��H@Q녿�  ?�=qA�  C��@Q��?��Az{C���                                    Bxߢu�  �          @���@k�����?G�A3�C��3@k����>��@��HC��\                                    Bxߢ�h  �          @��
@'
=�2�\���
�qC��@'
=�(��ٙ��˅C�w
                                    Bxߢ�  
�          @�(�@3�
�>�R�k��P  C���@3�
�5��z�H�\(�C�ff                                    Bxߢ��  �          @��
@/\)�@�׾�
=��ffC�.@/\)�333�����RC�/\                                    Bxߢ�Z  T          @~�R@���I��>��@qG�C��
@���G
=�
=q��(�C��                                     Bxߢ�   �          @�Q�@4z��4z���\C�~�@4z��.{�=p��*�HC�                                      Bxߢͦ  �          @��@6ff�9��>B�\@.{C�C�@6ff�6ff�����\)C�~�                                    Bxߢ�L  �          @��
@$z��E��G���{C��q@$z��8Q쿚�H����C�޸                                    Bxߢ��  �          @�
=@(��X�ÿ����t(�C�k�@(��@  ��
=��\)C�\                                    Bxߢ��  �          @�@5��<(��^�R�Ap�C��3@5��'��˅����C��R                                    Bxߣ>  �          @��\@G
=�#33��\)�|��C�P�@G
=����k��R�RC�0�                                    Bxߣ�  T          @��
@N�R�   >��H@���C�R@N�R�#33�.{��C�޸                                    Bxߣ%�  
�          @�p�@R�\�!�>�z�@���C�AH@R�\�!G���Q���Q�C�N                                    Bxߣ40  
�          @��
@<���6ff>���@�C�H@<���5���G���z�C�R                                    BxߣB�  
�          @�{@J�H�)����  �`��C��@J�H�   �k��NffC��                                    BxߣQ|  
�          @�(�@U����B�\�,(�C�(�@U�G��L���4  C���                                    Bxߣ`"  
�          @�p�@\(��
=�#�
�{C�@\(��\)�B�\�(��C�t{                                    Bxߣn�  �          @�(�@\(���Ϳ���33C��)@\(����R��=q�r=qC��                                    Bxߣ}n  	�          @�ff@P���'
=��
=����C���@P���=q��=q�n�HC���                                    Bxߣ�  
�          @��R@/\)�J=q=�\)?xQ�C�xR@/\)�Dz�@  �$��C��                                     Bxߣ��  
�          @�\)@ff�`  ���
��z�C��R@ff�W��n{�K
=C�t{                                    Bxߣ�`  �          @���@��_\)�#�
��C�k�@��W
=�p���L(�C��                                    Bxߣ�  �          @��H@z��g�>�33@��HC�U�@z��e�(��G�C�u�                                    BxߣƬ  �          @�(�@!��b�\>��
@�ffC���@!��`  �(�� ��C��3                                    Bxߣ�R  �          @��@
�H�p  �aG��>{C��@
�H�c�
��Q���C��
                                    Bxߣ��  �          @��\@���h��>\@�(�C��@���g��z����C���                                    Bxߣ�  �          @��\@
=�e>���@�33C���@
=�dz�����=qC���                                    BxߤD  �          @�(�@�\�mp���z��uC��3@�\�`  ��  ��p�C���                                    Bxߤ�  �          @�(�?�{�z�H���H����C��=?�{�i����  ���\C�t{                                    Bxߤ�  �          @�?�{�{��:�H�p�C��\?�{�fff�޸R��{C���                                    Bxߤ-6  �          @��?�p��\)��33����C���?�p��Z=q�(���ffC��{                                    Bxߤ;�  
�          @��?5�|�Ϳ���{C�  ?5�U��1��
=C��                                    BxߤJ�  �          @��\>Ǯ�y����ff�ř�C���>Ǯ�R�\�1G���C�XR                                    BxߤY(  �          @��
>aG��o\)��R���C��{>aG��AG��H���8
=C�q                                    Bxߤg�  �          @�(�?z��hQ�����\C���?z��8Q��Mp��>{C���                                    Bxߤvt  �          @�{?��
�Y���'
=�=qC�ff?��
�%��Z�H�Kp�C��3                                    Bxߤ�  �          @�  >\�h���'
=�  C���>\�4z��_\)�K=qC��
                                    Bxߤ��  �          @�
=>#�
�mp��{��C�7
>#�
�:�H�W��D33C���                                    Bxߤ�f  �          @�p���\)�i����R�ffC�p���\)�7
=�W
=�FG�C�H�                                    Bxߤ�  �          @�p�=L���fff�#33�ffC�c�=L���1��Z�H�K\)C��                                     Bxߤ��  �          @�33=L���\(��B�\�%��C�aH=L���   �vff�d
=C��f                                    Bxߤ�X  �          @�����U��J�H�.��C��3���ff�|���lC���                                    Bxߤ��  �          @�\)�����g��'��Q�C��)�����1��`  �M\)C��                                    Bxߤ�  �          @���\�e�!���C����\�1G��Z=q�I�\C��3                                    Bxߤ�J  
�          @�
=��
=�U��<���%ffC�k���
=�=q�o\)�cG�C�{                                    Bxߥ�  �          @�\)�!G��8���Vff�Bp�C�⏿!G������Q��  C{��                                    Bxߥ�  �          @�(��Y���(Q��Y���Kz�C|(��Y����{�~�R��CrQ�                                    Bxߥ&<  �          @��Ϳ0���.�R�XQ��H��C�=�0�׿��H�\)u�Cx                                      Bxߥ4�  �          @��Q��   �w
=�r�RCw�3�Q녿aG���Q�aHCc                                    BxߥC�  �          @�33�L���AG��A��0p�C33�L�����n�R�l��Cy
=                                    BxߥR.  �          @�33����?\)�<���*�Cy5ÿ�����i���d�CqQ�                                    Bxߥ`�  �          @�G���=q�"�\�<���-Q�CjT{��=q����b�\�\�C]��                                    Bxߥoz  �          @��
�p��$z��0  �(�Ce8R�p���(��U�H33CY�\                                    Bxߥ~   �          @�(���(��Vff��
��Cr�Ϳ�(��)���9���*
=Cm�                                    Bxߥ��  �          @��\���p  ��\)���CyE���J�H�%���Cu�{                                    Bxߥ�l  �          @����z��c33�ff���HCx\)��z��5��@  �-�
Cs�{                                    Bxߥ�  �          @�(���(��_\)�z�� z�Cz�Ϳ�(��.{�L(��;��Cu�f                                    Bxߥ��  T          @����
=�E��,(��z�CuͿ�
=�{�\(��QCm&f                                    Bxߥ�^  �          @������E��*�H�ffCsY�����{�Z�H�NCk(�                                    Bxߥ�  �          @�(���=q�1G��G
=�5G�Ct^���=q��ff�p  �l�Ci��                                    Bxߥ�  �          @��H��Q��2�\�4z��#�Cn�q��Q��33�_\)�W��Cd@                                     Bxߥ�P  �          @�G��޸R�+��6ff�&�
Cm{�޸R����^�R�Y�Ca�\                                    Bxߦ�  �          @�녿У��'
=�>{�0�Cn
�У׿�
=�e��cz�Ca�3                                    Bxߦ�  �          @��
��\)��
�`  �T�Cg�H��\)��G��{�G�CS�H                                    BxߦB  �          @��Ϳ���
=�dz��Y�Ce\���Y���~{ffCO                                    Bxߦ-�  �          @����Ϳ�G��u�p�
C_8R���;�p����\)CA
=                                    Bxߦ<�  �          @��\��������r�\�u�Cf�����������HW
CG޸                                    BxߦK4  �          @��\���H�����z�H�fCdǮ���H�aG�����
=C>p�                                    BxߦY�  �          @��H�+���G��~{�3Cv��+���{��\)�fCN��                                    Bxߦh�  �          @�(���(���=q��ff�CxL;�(�>�����H¨
=C �{                                    Bxߦw&  T          @�p�����������Q�C~������>������«=qC                                      Bxߦ��  "          @��
�����z���ffz�C��=���=��
���¬�)C#)                                    Bxߦ�r  �          @��>�=q��=q���C��=>�=q��Q���(�¬�=C��                                    Bxߦ�  �          @�z�#�
�\���8RC�{�#�
��z����
«aHCq޸                                    Bxߦ��  �          @�(������R�s33�v�RC�:���Tz����RffCo�                                    Bxߦ�d  �          @������   �S�
�F
=CrJ=��������xQ��|�HCc��                                    Bxߦ�
  �          @��H�@  ��p��qG��r\)Cy=q�@  �Tz���p�Cc�R                                    Bxߦݰ  �          @�\)��  �=q�R�\�NG�Cw����  ��\)�uaHCi�3                                    Bxߦ�V  �          @�Q쿀  �<���4z��(�C{33��  �G��c33�f\)Cs�=                                    Bxߦ��  �          @��\�aG��P���3�
�"�C�;aG��z��hQ��effC�B�                                    Bxߧ	�  �          @�  �!G��W���R���C��{�!G�� ���Vff�P��C�                                    BxߧH  T          @�녿=p��Tz��#�
��C��
�=p�����Y���T\)C}#�                                    Bxߧ&�  �          @��H���
�b�\�   �ffC�s3���
�*�H�Z=q�O(�C���                                    Bxߧ5�  
Z          @�녾��R�E��>{�/{C�#׾��R��n�R�q�
C��                                    BxߧD:  �          @��þ��:�H�Dz��7�
C�O\����z��q��y�HC�\                                    BxߧR�  �          @�=q���X���+���\C�����p��c33�]  C���                                    Bxߧa�  �          @��H�(���P���1G���C�K��(����
�fff�aG�C~)                                    Bxߧp,  �          @��H��z��J=q�<���+p�C�l;�z��
�H�o\)�n��C�AH                                    Bxߧ~�  �          @����{�8Q��N�R�@  C��)��{�����{��\C���                                    Bxߧ�x  �          @�33�
=q�>{�B�\�4��C��H�
=q�����q��w33C~�H                                    Bxߧ�  �          @��\�:�H�E��=p��,G�C�Uÿ:�H���n�R�m��Cz�f                                    Bxߧ��  T          @��\�
=q�8���K��<�RC��\�
=q��=q�x���33C}��                                    Bxߧ�j  �          @�������1��N{�C=qC�'���׿��H�y��
=C~z�                                    Bxߧ�  �          @��H���&ff�[��Q=qC�������p���G��fC{ٚ                                    Bxߧֶ  �          @�=q��\)�7
=�N{�A�C�8R��\)���
�z�H��C��                                    Bxߧ�\  �          @��\��\)�*=q�XQ��Op�C�4{��\)��������C��                                     Bxߧ�  �          @�녿   �Q��.�R��HC��{�   �z��e��aC���                                    Bxߨ�  �          @��\����HQ��9���)(�C��{����Q��l���lz�C~G�                                    BxߨN  �          @�33�(���-p��J=q�B=qC�&f�(�ÿ�33�tz�.Cx&f                                    Bxߨ�  �          @�������C33�Fff�6�C�����Ϳ�p��w��{ffC���                                    Bxߨ.�  �          @�����R�<(��L���=Q�C��q���R�����|(��C�=q                                    Bxߨ=@  �          @����G��:=q�R�\�@�
C�����G������Q���C�                                    BxߨK�  �          @�z����0���W
=�GC�箿���У�����z�Cy�)                                    BxߨZ�  �          @���8Q��+��W��I��C
=�8Q��ff������Cu&f                                    Bxߨi2  �          @����n{�,(��N�R�Bz�Cz�ÿn{�˅�x��B�Co�=                                    Bxߨw�  �          @����O\)�#�
�W
=�Mp�C|}q�O\)��
=�~{�Cp�                                    Bxߨ�~  �          @��H�E��33�e�_�HC{aH�E���{���
�Ck�                                    Bxߨ�$  T          @�(��.{��\�|(���Cx�ÿ.{���H��G���CX�                                    Bxߨ��  �          @����0�׿W
=��z��Cf��0��>���ff�RCW
                                    Bxߨ�p  �          @�녿h�ÿ����x��
=Co��h�þ��R��(�CG
                                    Bxߨ�  �          @������׿&ff���\��CQ�Ϳ���?�R���\�fC\                                    Bxߨϼ  �          @�  ��G���ff��Q�
=CG�=��G�?J=q�}p�ǮC�                                    Bxߨ�b  �          @�Q쿂�\��G��fff�rQ�Co޸���\���}p�#�CQQ�                                    Bxߨ�  �          @�=q�(���5�I���<��C�lͿ(�ÿ޸R�w
=� Cy#�                                    Bxߨ��  �          @�=q�u�!��Vff�L\)CyG��u�����}p�\CkG�                                    Bxߩ
T  �          @�  �z�H�/\)�E�:�CzY��z�H��z��q��|�Cou�                                    Bxߩ�  �          @����G��-p��O\)�@��Cys3��G���=q�z=qCmG�                                    Bxߩ'�  �          @�녿�  �B�\�8Q��'�C{��  �   �j�H�j��Csp�                                    Bxߩ6F  �          @�������E�333�"�C{�������g��d��Cs�                                    BxߩD�  T          @�33��=q�(���L(��<CsE��=q�\�vff�y=qCd޸                                    BxߩS�  �          @��H��=q�E�333�"{Cz���=q�z��g��d�Cr�                                     Bxߩb8  �          @��\��(���R�X���O(�Cl� ��(�����z=q8RCW�=                                    Bxߩp�  �          @��H�=p��+��R�\�G  C~xR�=p��\�}p���Cs�3                                    Bxߩ�  �          @�(��O\)���e�\G�C{ͿO\)����������CjQ�                                    Bxߩ�*  �          @��Ϳ0���#�
�_\)�SG�C~�3�0�׿�����
�
Cr�\                                    Bxߩ��  �          @�zᾨ���1G��XQ��I��C��q���ÿ������\
=C�R                                    Bxߩ�v  �          @�33���H�*�H�X���M\)C��f���H��(�����.C{}q                                    Bxߩ�  �          @����c�
���`  �Z(�Cy��c�
��{����G�Cg+�                                    Bxߩ��  �          @�����R���_\)�UCq� ���R�������  C\�f                                    Bxߩ�h  �          @��\���
�Q��Z�H�Q��CjG����
�p���z�H��CS��                                    Bxߩ�  �          @����ٙ��
=q�S�
�IQ�Cg��ٙ���  �tz��z�CR��                                    Bxߩ��  �          @�=q��  ��\�XQ��M��CeLͿ�  �\(��vff�|\)CN:�                                    BxߪZ  �          @��\�˅�p��W
=�L{CjW
�˅���
�x��#�CT�q                                    Bxߪ   �          @�33����{�N�R�@
=Cn)�������vff�y�C\�=                                    Bxߪ �  �          @��
��ff���X���KCe@ ��ff�fff�xQ��z�CN�                                    Bxߪ/L  �          @��
��  �����dz��[
=Cb!H��  �z��~{�CFz�                                    Bxߪ=�  �          @�33��p���Q��mp��k
=C[��p��8Q��~�R�qC9޸                                    BxߪL�  �          @��Ϳ˅��ff�x���x��C[B��˅<��
��33�RC3T{                                    Bxߪ[>  �          @��
��=q��33�mp��i33CbJ=��=q�\��G�(�CAp�                                    Bxߪi�  �          @�(���
��Q��e��[��CV���
�W
=�vff�v��C9�{                                    Bxߪx�  �          @����ff��
�O\)�?{C`p��ff�h���n�R�i��CKQ�                                    Bxߪ�0  �          @�ff��p���R�R�\�?z�Cdc׿�p���ff�u�o  COٚ                                    Bxߪ��  T          @��R��\)�(��Y���F�HCe���\)�z�H�{��w  CO��                                    Bxߪ�|  �          @���   ��z��^{�T��C[���   ��ff�tz��w�C@��                                    Bxߪ�"  �          @�p��G��"�\�=p��)z�Cg�{�G������hQ��^z�CW�)                                    Bxߪ��  
�          @��R�G���\�Dz��.�CaJ=�G����i���[�HCOJ=                                    Bxߪ�n  �          @�p��G��p��Dz��0p�C`5��G�����g��\�
CM�
                                    Bxߪ�  �          @�p��  �ٙ��S33�F(�CY&f�  ���k��g��CAE                                    Bxߪ��  �          @���   �fff�`  �S=qCG�   >����fff�\33C-@                                     Bxߪ�`  �          @�p���Ϳ8Q��fff�Z�RCDs3���?   �h���^(�C(��                                    Bx߫  �          @����\��\�n�R�g  C@�H��\?=p��l(��c\)C"+�                                    Bx߫�  �          @����ÿ8Q��n�R�kQ�CF�����?
=q�p���nffC%��                                    Bx߫(R  �          @�  �
=�^�R�w��n  CJ}q�
=>�ff�|���u\)C'��                                    Bx߫6�  
�          @�Q����z�H�tz��g�CLJ=��>����{��r�RC+\)                                    Bx߫E�  �          @����zῂ�\�y���c�HCK�3�z�>�����Q��n�C+�3                                    Bx߫TD  T          @�p���\��(���Q��mffCS  ��\>.{��{��C/�                                    Bx߫b�  �          @�z���
��=q�|(��hz�CT�3��
=L������~C2�{                                    Bx߫q�  �          @�p�����G��y���b�\CX��������~(�C7�                                     Bx߫�6  
�          @����\�xQ��a�CX��\)����}Q�C7�                                    Bx߫��  �          @��
=��33�u��\Q�CZ  �
=��=q����{C;aH                                    Bx߫��  �          @�ff��R��
=�qG��U��CX���R���
���
�u(�C<!H                                    Bx߫�(  �          @��H�����(��o\)�Z=qCc�׿�������\)CFz�                                    Bx߫��  �          @�33�ff��ff�j=q�S�C\��ff��������x=qC@��                                    Bx߫�t  �          @�(��
�H��
=�mp��V{CYǮ�
�H��������vp�C<�                                    Bx߫�  �          @���z��
=�]p��C�
C[�R�z�&ff�z=q�i�CC�{                                    Bx߫��  �          @�(����G��aG��I=qCX�����z=q�j�C?5�                                    Bx߫�f  �          @�z��33� ���mp��U
=Cb�H��33�#�
����CF��                                    Bx߬  �          @�33�ff��`  �F��C`�)�ff�G���  �s  CH^�                                    Bx߬�  �          @��
��
�p��U��8��C_�
��
�s33�xQ��e�CJp�                                    Bx߬!X  �          @�33�33���QG��5=qC`���33���
�vff�cG�CL)                                    Bx߬/�  �          @�33�
=��L(��/=qC`�=�
=��{�r�\�^  CMJ=                                    Bx߬>�  T          @����  �(��G��,�CcT{�  ��p��p���_
=CP�H                                    Bx߬MJ  �          @��\�{� ���E�*�\Cd���{����p���^ffCRs3                                    Bx߬[�  �          @��\�!��{�9�����C`B��!녿����dz��Mp�CO�{                                    Bx߬j�  �          @����\)�&ff�>{�#p�CeJ=�\)��
=�l(��X�HCTu�                                    Bx߬y<  �          @�G���)���A��'��Cg� ������p  �_��CV�3                                    Bx߬��  �          @�Q�����*�H�/\)�\)Cd�����Ǯ�_\)�J��CU�                                    Bx߬��  �          @�ff�=q�*�H�#�
���Cc���=q��\)�Tz��C�CU�{                                    Bx߬�.  �          @���AG��!G���p��ծC[�{�AG���
=�.�R�(�CQ�                                    Bx߬��  �          @�p��-p��/\)��\����Ca\)�-p���\)�7
=�$(�CV�\                                    Bx߬�z  �          @���G����\���D��C\O\�G��(��z=q�k�CB��                                    Bx߬�   �          @���(��Q��Vff�7��C]&f�(��W
=�xQ��a�CF�f                                    Bx߬��  �          @��\)��
�N{�-C^���\)����tz��[
=CJ�=                                    Bx߬�l  T          @��   ���J=q�*�C_u��   ��\)�r�\�X�\CL�                                    Bx߬�  �          @�{���%�C33�"�Cbٚ����{�qG��VQ�CQ5�                                    Bx߭�  �          @���p��%��@��� �RCbh��p���{�n{�S�
CQ�                                    Bx߭^  �          @�ff�%��"�\�@  ��C`}q�%������l���O�CO!H                                    Bx߭)  �          @��R�#�
�&ff�>�R��CaaH�#�
�����l���O=qCPY�                                    Bx߭7�  �          @�p��\)�'
=�=p���CbW
�\)��33�l(��Q
=CQQ�                                    Bx߭FP  �          @�(����333�7
=���Cf=q����{�j=q�QQ�CV��                                    Bx߭T�  �          @��H����?\)�/\)���Cjh���ÿ����g��PC\xR                                    Bx߭c�  �          @�����C�
�%��Ck�����Q��`���K�C^�f                                    Bx߭rB  �          @����z��C33�'���RCkǮ�z���a��Mp�C^�=                                    Bx߭��  �          @�Q��Q��L(��!��z�Cn���Q����`  �Kz�Cc                                      Bx߭��  �          @�G��z��5�*�H�
=Cf�{�z���H�`  �JffCXc�                                    Bx߭�4  �          @�G��=q�.�R�-p���Cd�
�=q�˅�`���J33CUW
                                    Bx߭��  �          @��
� ���*=q�3�
�Q�Cb�H� �׿�p��e��Kp�CRs3                                    Bx߭��  
�          @�p��(���   �;��z�C_�=�(�ÿ���hQ��LQ�CN�                                    Bx߭�&  �          @�z��%��$z��7����C`�H�%���\)�fff�K�CO�R                                    Bx߭��  �          @���"�\�3�
�)�����Cc���"�\���^�R�C�CUY�                                    Bx߭�r  �          @�=q�%�)���+��p�Ca���%��G��\���D=qCR8R                                    Bx߭�  �          @���"�\�"�\�4z��Q�Ca��"�\��{�b�\�K��CP)                                    Bx߮�  �          @�Q��'
=��
�6ff�=qC]���'
=��\)�`  �K�CKT{                                    Bx߮d  �          @���*=q����4z���\C\T{�*=q����\���H�\CJ33                                    Bx߮"
  �          @�  �(Q��
�H�:�H�"p�C[���(Q�u�`���M�
CH+�                                    Bx߮0�  �          @���1G���\�8��� (�CXk��1G��Y���[��Gp�CE)                                    Bx߮?V  �          @�\)� ������333���C`E� �׿�G��`  �M�CN�H                                    Bx߮M�  �          @�\)����<����R���Cj���ÿ��X���H��C\�                                    Bx߮\�  �          @����p��8���2�\�  Ck�)��p����i���Z�C\5�                                    Bx߮kH  �          @�{��=q�A��*�H��Cn녿�=q���e�X�CaB�                                    Bx߮y�  �          @������?\)�0����
Cm�׿�׿�\�j=q�[33C_\)                                    Bx߮��  �          @�
=���
�\)�L���9�Cjk����
����x���u�CT�)                                    Bx߮�:  �          @�p���{��H�S�
�B��Clc׿�{����~{u�CT�                                     Bx߮��  �          @��
��z��	���g
=�_Cq�f��z�(����z�G�CQ�H                                    Bx߮��  �          @����G���H�R�\�D=qCn��G����
�|��B�CVff                                    Bx߮�,  �          @�p��˅�  �\���Mp�Cj�=�˅�Q���G�=qCO5�                                    Bx߮��  �          @����H�33�aG��S=qCf
���H�������� CG@                                     Bx߮�x  �          @������Q��[��K�Ce޸����333�~{=qCIk�                                    Bx߮�  �          @��Ϳ�p����W
=�G(�Ca���p��#�
�w��v��CE�)                                    Bx߮��  �          @�z��(���=q�^{�P��C^����(�����y���z�C?                                    Bx߯j  �          @����(���{�fff�Z�
C[J=��(�����|(��}p�C8s3                                    Bx߯  �          @�p���׿�Q��n{�e�HCYaH���=L���~�RffC2W
                                    Bx߯)�  �          @�33���ÿǮ�g
=�a�C\�����ý����{�B�C7�                                    Bx߯8\  �          @�  ����>�R�<(��#G�Cq
��녿�z��u�j��Ca=q                                    Bx߯G  �          @�{�����8Q��4z���Cm�����ÿ�{�l���bC]s3                                    Bx߯U�  �          @�ff���*=q�Dz��.  CkJ=������u�n=qCWaH                                    Bx߯dN  �          @�  ��33�+��Mp��6G�CnT{��33���\�~�R�y�CY�\                                    Bx߯r�  �          @�녿��H���^{�G��Cj=q���H�aG�����
COaH                                    Bx߯��  �          @�Q��\)�&ff�S33�<�RCnͿ�\)��33��G��(�CWz�                                    Bx߯�@  �          @������=q�Q��<G�Ch
=��׿z�H�|���w�CO��                                    Bx߯��  �          @�\)���H�\)�R�\�=�Ck�=���H����\)�|�HCS�q                                    Bx߯��  �          @����=q�p��Y���E�Cm8R��=q�z�H���\\CS��                                    Bx߯�2  �          @��R��z�� ���S33�>�\Clff��zῇ���  �
=CT}q                                    Bx߯��  �          @�\)��p��)���J�H�4{Cl�
��p����R�|(��vffCW��                                    Bx߯�~  �          @�Q�˅�"�\�XQ��Ap�Cm�3�˅������H�HCU�f                                    Bx߯�$  �          @�G���p��   �_\)�H�Co^���p��xQ����CUO\                                    Bx߯��  �          @�Q쿹�����`���LffCo.�����fff���CS�\                                    Bx߰p  �          @�Q�ٙ��G��]p��J�\CiE�ٙ��E���=qCLn                                    Bx߰  �          @�Q��33�=q�\(��F��Ck���33�fff��33��CP��                                    Bx߰"�  �          @���ٙ��ff�Z�H�F�Cj
�ٙ��Y������HCN�                                     Bx߰1b  �          @�
=�\�	���g��WCj�׿\�z���p�33CH��                                    Bx߰@  �          @�p�������g
=�[33C`� ���������#�C<
                                    Bx߰N�  �          @����ÿ�\�o\)�fCd^����þ8Q���(��C:�\                                    Bx߰]T  �          @��������l���b��Cf�\�������������C>�R                                    Bx߰k�  �          @��Ϳ�
=����h���`{Cb����
=�k�����z�C;Ǯ                                    Bx߰z�  �          @�
=��
=��
=�j�H�\G�C]���
=������ǮC7��                                    Bx߰�F  T          @�\)��G����p���d�
C_}q��G���\)���
�3C68R                                    Bx߰��  �          @������G��p  �d(�Cbz���#�
��z�ǮC9ff                                    Bx߰��  �          @�ff�\�z��\���L��Cl�{�\�J=q���\� COu�                                    Bx߰�8  �          @����Q��=q�\���PG�Cs��Q�^�R���
�3CX
                                    Bx߰��  �          @�����  �\(��K��Cj���녿8Q�����L�CK�{                                    Bx߰҄  �          @�{��{�  �U�Cz�Cfn��{�@  �}p��|p�CJ{                                    Bx߰�*  �          @�ff������{�a��R�RC_�q�������
�~�R�~��C=T{                                    Bx߰��  �          @��
��{�z��U��F��Ck@ ��{�Q��~{\CO�                                    Bx߰�v  �          @�z�Ǯ�{�QG��A  Cm�=�Ǯ�z�H�~�Ru�CT)                                    Bx߱  �          @��Ϳ��.{�J=q�7Q�Crn����  �~�R(�C]L�                                    Bx߱�  �          @��Ϳ���8Q��HQ��5
=Cxff��녿�z���Q�p�Cf�                                    Bx߱*h  �          @�(���=q�9���E�3�RCy}q��=q��Q��\)k�Ch�                                    Bx߱9  �          @�(���33�<(��A��/z�Cx�
��33���R�|��\ChE                                    Bx߱G�  T          @��Ϳ�z��7
=�G��5ffCw�ÿ�z῰����Q��\Ce��                                    Bx߱VZ  �          @�p���=q�,���R�\�B�Cx#׿�=q��z���33��Cc                                    Bx߱e   �          @�
=�������[��J
=Ck�)���ͿE����\CM��                                    Bx߱s�  �          @�p���  ���U�D�HCh����  �B�\�~{�3CKff                                    Bx߱�L  �          @����ff����U��D33Cg\)��ff�:�H�}p��(�CJ
                                    Bx߱��  �          @��ٙ��Q��^�R�Oz�CgW
�ٙ��\)��G�#�CF�                                    Bx߱��  �          @�ff����R�dz��U�Cmuÿ��(����33CK#�                                    Bx߱�>  �          @����=q���^�R�Q{Cj\��=q�
=��=q�HCH��                                    Bx߱��  �          @�(���ff�   �e��ZCh.��ff�Ǯ���\#�CB�                                    Bx߱ˊ  �          @��
��\�   �S33�D�C`T{��\���H�u��u  CA}q                                    Bx߱�0  �          @�p��
�H��\)�Vff�E��C\�q�
�H��33�tz��p=qC=E                                    Bx߱��  �          @���\��
�K��7C^
��\����p  �g
=CB��                                    Bx߱�|  �          @����\����?\)�+�RC`����\�Y���i���`��CHQ�                                    Bx߲"  T          @�����H��
�6ff�!�RC_����H�s33�b�\�W  CI�=                                    Bx߲�  �          @�p���\�#33�1G��Cd
��\�����dz��XG�CO��                                    Bx߲#n  �          @����ff�8���� ��Cf�H�ff��
=�S�
�C�RCW�)                                    Bx߲2  �          @��H��,(��*�H��Ch.���{�a��[Q�CU#�                                    Bx߲@�  �          @����
=��R�L���=�Ce&f��
=�:�H�u��v�RCH��                                    Bx߲O`  �          @�=q�Ǯ���c33�_=qCe�Ǯ�aG���  ��C<33                                    Bx߲^  �          @����p���  �s�
�y\)CR33��p�?&ff�xQ�B�Cn                                    Bx߲l�  �          @����\)��33�w��{�CW=q��\)?
=q��  ��C!��                                    Bx߲{R  �          @�=q�\)�G��7��({Ca^��\)�aG��c33�_p�CI�\                                    Bx߲��  �          @��H�Q���\�?\)�.��Cc��Q�Y���j�H�g��CI�{                                    Bx߲��  �          @��
�	�����A��0  Cb���	���Q��l���h33CH�3                                    Bx߲�D  �          @�33�Q쿴z��^�R�V��CUu��Q�>#�
�p  �q=qC/��                                    Bx߲��  �          @��
�   ���XQ��JG�C_�=�   ��33�w��y
=C=�                                    Bx߲Đ  
�          @�z��\)����5��#=qCb�=�\)��  �c�
�]�RCL                                    Bx߲�6  �          @��
�*=q� �������C_n�*=q����N{�>
=CM�                                    Bx߲��  �          @�33�.�R���(��	�
C[��.�R����J�H�=�CI=q                                    Bx߲��  �          @�=q�녿�z��P  �EffC_E�녾\�p  �t�C>��                                    Bx߲�(  �          @��
��H��Q��C33�1��CZ����H���H�e��]z�C?p�                                    Bx߳�  �          @��\�z����K��=�CY���zᾔz��h���d��C;+�                                    Bx߳t  �          @��\�#33��=q�G��8�CS���#33���`  �W\)C6�                                    Bx߳+  �          @�G��@  ����0  �   CK��@  �#�
�B�\�5z�C4��                                    Bx߳9�  �          @�Q��HQ쿓33�(���Q�CH&f�HQ�=u�8Q��*=qC2�{                                    Bx߳Hf  �          @��H�<�Ϳ��\�:�H�)=qCKG��<��=��
�K��<C2�                                     Bx߳W  �          @�(��0  ��=q�@  �-��CQ��0  �.{�X���K�\C7�
                                    Bx߳e�  �          @�z��-p���{�@���.�RCR���-p��B�\�Z=q�M��C8�                                    Bx߳tX  �          @����:=q���R�0��� �CO
=�:=q�8Q��HQ��<33C7�=                                    Bx߳��  �          @��H�9���Ǯ�5��"�CP:��9���aG��N{�?�RC8=q                                    Bx߳��  �          @���<���
=��ffCW���<�Ϳk��@���1�CEJ=                                    Bx߳�J  �          @��\�333�ff� �����CX�)�333�W
=�J�H�=�CD�H                                    Bx߳��  �          @��\�0���ff�&ff��\CYL��0�׿L���P  �B{CD�                                    Bx߳��  �          @�33�0  ��z��6ff�&z�CS&f�0  ��z��R�\�G�
C:�                                    Bx߳�<  �          @��
�>{�W
=�?\)�0=qCCٚ�>{?�\�C�
�5�HC*E                                    Bx߳��  �          @����HQ쿧��'
=�G�CJ�3�HQ콏\)�:�H�,  C5O\                                    Bx߳�  
�          @��H�>�R��33�333�!ffCM)�>�R��\)�G��9ffC5Q�                                    Bx߳�.  �          @�33�Fff����,����
CL5��Fff��G��B�\�1�\C5�3                                    Bxߴ�  �          @�z��>�R��p��7
=�"��CNT{�>�R��G��N{�<��C6{                                    Bxߴz  �          @�{�:�H���+��(�CUQ��:�H�z��P  �>  C?G�                                    Bxߴ$   �          @�z��333�\�@���-p�CPz��333���
�W��I
=C5��                                    Bxߴ2�  
�          @����)����z��Mp��;�CP��)��=����`  �S�\C1�=                                    BxߴAl  �          @�z��+���33�:=q�%�CWO\�+����\(��N��C=��                                    BxߴP  �          @�(��>�R�����*=q���CS���>�R���L���:\)C=��                                    Bxߴ^�  �          @�(��333���G����
C\(��333��33�Dz��5\)CJaH                                    Bxߴm^  �          @�(��1G��G��\)�
=C[J=�1G��xQ��N{�>��CGJ=                                    Bxߴ|  �          @�33�n�R�У׿��R���\CK�H�n�R�Tz��33��p�C@�                                     Bxߴ��  �          @��\�j�H��׿�Q���Q�CO#��j�H��
=�����Q�CE�
                                    Bxߴ�P  �          @��
�vff��G����
�[
=CL�\�vff��녿�Q����HCDu�                                    Bxߴ��  �          @�z��j�H�zῐ���p��CQxR�j�H��\)����
=CH�=                                    Bxߴ��  �          @���u���G��J=q�(��CL��u����R��(����HCE�R                                    Bxߴ�B  �          @�ff��녿�\��\)�e�CK�)��녿�p����\�U�CG�q                                    Bxߴ��  �          @�p����H��33>B�\@��CJ
=���H���ÿ����{CI�                                    Bxߴ�  T          @�p���  ��\��\��{CK����  ��\)���R��ffCF�f                                    Bxߴ�4  �          @����|(��������|(�CM�3�|(��˅��{�jffCJ                                      Bxߴ��  �          @���}p����fff�8z�CMٚ�}p���=q��33��(�CF�{                                    Bxߵ�  �          @�Q��}p����Ϳ�{�c�CM\�}p���
=����
=CD�\                                    Bxߵ&  �          @�G��~{��
�.{�
�\COu��~{��ff�\���HCIaH                                    Bxߵ+�  �          @��\����G��(����CNk���녿\��p���ffCH�\                                    Bxߵ:r  �          @�33��(���녿B�\��
CL�)��(���{��G����CFG�                                    BxߵI  �          @�33���׿�p��}p��G�
CN5����׿����G�����CFh�                                    BxߵW�  �          @��
������
�}p��EG�CO
���׿�z������RCGO\                                    Bxߵfd  �          @���\)���}p��FffCO���\)��
=������CG��                                    Bxߵu
  �          @�����  �33��
=�k33CO
��  ������H���CF(�                                    Bxߵ��  �          @�=q�x���33���\���RCOǮ�x�ÿ��
�33��ffCF&f                                    Bxߵ�V  T          @�
=���
�ٙ�����=qCJn���
��(��^�R�2�HCG�H                                    Bxߵ��  �          @������H��\)�&ff��RCL�=���H��녿�z���z�CF��                                    Bxߵ��  �          @���~�R��p��}p��I��CNz��~�R��=q��\��33CF�                                    Bxߵ�H  �          @����w����33�k33CPT{�w����Ϳ��H�̣�CGG�                                    Bxߵ��  �          @����p����������CR(��p�׿�������
=CG�\                                    Bxߵ۔  �          @����r�\�	�����R�}G�CQ�{�r�\��\)�z����CG��                                    Bxߵ�:  �          @�Q��k��Q쿎{�b�RCT�{�k��У��33��{CK޸                                    Bxߵ��  �          @�  �i���������c�CU��i���У��33��G�CL�                                    Bx߶�  �          @�Q��e�����
=�u�CU���e��\)�Q����
CL@                                     Bx߶,  �          @����\���p��������CW���\�Ϳ�p��"�\�=qCK=q                                    Bx߶$�  �          @����b�\�p���G���33CT�b�\��
=�$z��
33CF��                                    Bx߶3x  T          @����o\)��ÿ����z�CQ�o\)����
=q����CG=q                                    Bx߶B  �          @��R�s�
��\)������p�CN33�s�
����33��p�CC��                                    Bx߶P�  
�          @����p�׿�Q쿚�H�\)CO@ �p�׿�
=�������HCEu�                                    Bx߶_j  �          @���mp���\�����|Q�CP�{�mp����
���R��
=CG\                                    Bx߶n  �          @�p��c�
�������g
=CU�f�c�
��\)�33��CLh�                                    Bx߶|�  �          @����\��� �׿�=q�c33CX��\�Ϳ�  �ff��{CN�                                    Bx߶�\  �          @�(��XQ��\)��p�����CXn�XQ����R����CN:�                                    Bx߶�  �          @�(��S�
�!녿�\)��CYW
�S�
�У����CN0�                                    Bx߶��  �          @���W
=�#33�����{CY33�W
=��
=��
��z�CN�H                                    Bx߶�N  �          @�33�XQ��(���p����CW�H�XQ��\)�p���(�CM��                                    Bx߶��  �          @�=q�W
=�����
����CW8R�W
=���
�{��z�CLp�                                    Bx߶Ԛ  �          @����K��1녿333�33C]#��K��	��������CV{                                    Bx߶�@  �          @����Z�H�p��O\)�.=qCW�q�Z�H���ÿ�����CP�                                    Bx߶��  �          @���Z=q���������\CVE�Z=q�����������CK�                                    Bx߷ �  �          @����XQ����G���{CV�f�XQ��  �(����RCK�3                                    Bx߷2  �          @�  �Y���z῅��a�CVW
�Y���˅��p�����CM\                                    Bx߷�  �          @�Q��N{�%��  �ZffCZ޸�N{������z�CQ�\                                    Bx߷,~  �          @�  �J=q�-p��W
=�733C\���J=q�G����R��  CT��                                    Bx߷;$  �          @���J=q�2�\����C]u��J=q�G���Q���CW�                                    Bx߷I�  �          @�
=�J=q�)���E��)�C\�J=q�   ��z����CTW
                                    Bx߷Xp  T          @��AG��5��Q����C_J=�AG��
=�У���  CZ�                                    Bx߷g  �          @��333�E=��
?�\)Cc�{�333�1녿������C`�=                                    Bx߷u�  �          @�{�8Q��A녾8Q��(�Cb���8Q��'
=�Ǯ����C^0�                                    Bx߷�b  �          @��0  �HQ�8Q��=qCd���0  �,(��������RC`^�                                    Bx߷�  �          @�����N{�W
=�<��CiY�����p���R�{Ca��                                    Bx߷��  �          @�{��33�QG����
��\)Co�H��33�z��E�>�
Cc��                                    Bx߷�T  �          @�p�����R�\��R��Ci�����'������Cc��                                    Bx߷��  �          @�����Z�H����g33Cm�����"�\� ���{Ce�                                    Bx߷͠  �          @�(���Q��\(�����z�Cp����Q��   �(���\)Ch8R                                    Bx߷�F  �          @����p��^{���H���CsxR��p�����:=q�2G�Cj�                                    Bx߷��  ,          @�z��=q�c�
�u�W33Cr�Ϳ�=q�-p�� ���
=Ck��                                    Bx߷��  T          @\)�G��A�?�\)A��Ci��G��L�;�
=��Cj�H                                    Bx߸8  �          @x�ÿ�Q��;�?���A�=qCl���Q��U�<��
>k�CoǮ                                    Bx߸�  �          @����  �QG�?+�A�CkxR�  �Mp��p���W�Cj�                                    Bx߸%�  �          @�p��
=q�c33>#�
@G�Cn���
=q�N{�\����Cl
                                    Bx߸4*  �          @�\)���^�R�W
=�5�Ck�3���>{�����
Cgn                                    Bx߸B�  �          @�
=��
�XQ�n{�Lz�Ck����
�#33�=q��HCc�)                                    Bx߸Qv  T          @����
=�QG���(���
=Cj:��
=��
�'���C`xR                                    Bx߸`  �          @�G����U��Q���z�Cn!H���	���C�
�6�
Cb                                    Bx߸n�  �          @�\)� ���N�R������z�Cn33� ���   �HQ��?p�C`��                                    Bx߸}h  �          @�\)�
=q�:=q����  Ci���
=q����R�\�L��CW�H                                    Bx߸�  �          @�Q�����G���z���{Ck����ÿ�{�J=q�@z�C\�3                                    Bx߸��  �          @�����E��p��ߙ�Cl�������L���Ep�C\��                                    Bx߸�Z  �          @�\)��p��QG���p���=qCn���p��z��Dz��;�
CbB�                                    Bx߸�   T          @�\)���7
=����{Cm5ÿ���{�`���c�\CXs3                                    Bx߸Ʀ  T          @���G��?\)�(���z�Ck�R�G��˅�Vff�Q�
CZO\                                    Bx߸�L  T          @�ff��7������z�Cf����У��AG��:CV�)                                    Bx߸��  T          @�{����G
=��p���p�Cj�R��Ϳ��@  �7�C]�                                    Bx߸�  T          @���\�Y����ff���RCo)��\�Q��0���%\)Ceu�                                    Bx߹>  T          @����H�W
=�\��33Co� ���H��R�;��2Cd��                                    Bx߹�  �          @�����X�ÿ�G��c�
Cnh����   �   ��Cf0�                                    Bx߹�  T          @�������J=q�����\Cjp�����
=�0���&�C^�3                                    Bx߹-0  �          @��
���?\)��ff��\)Cg�\������2�\�*��CZ}q                                    Bx߹;�  T          @�ff�#�
�5��p���Q�Cd
=�#�
���7��,�CU=q                                    Bx߹J|  �          @����=q�c33��G��iCu�q��=q�(Q��%�� G�Co
=                                    Bx߹Y"  "          @��
��z��y�����H��C}h���z��L(��33�p�Cy��                                    Bx߹g�  
�          @����G��r�\�aG��G�C{�H��G��:=q�%��Cv�{                                    Bx߹vn  "          @��
����i������s\)Cy33����,���+��%�Cr�                                    Bx߹�  T          @�z��G��^�R�Ǯ����Cv���G���
�B�\�?{Cl�                                     Bx߹��  
�          @�(�����aG��xQ��[\)Cq�����'��#33��CjB�                                    Bx߹�`  �          @�(���z��dz�@  �'
=Cq�\��z��1G����z�Ckc�                                    Bx߹�  
(          @�33��
=�\(������r=qCp�3��
=� ���%�Q�Ch\)                                    Bx߹��  
�          @��
���H�O\)�˅��Q�Cn�)���H�z��<���8  Cb��                                    Bx߹�R  ,          @��
��\�N{����ffCq0���\��Q��H���H33Cc�R                                    Bx߹��  	�          @�33���R�Y��������=qCvY����R�p��$z��'(�Cn޸                                    Bx߹�  T          @�����hQ�c�
�I�Cu8R���/\)�!���Cn�H                                    Bx߹�D  T          @����   �Y����(�����Co���   ����-p��$(�Cf
=                                    Bxߺ�  �          @�{�
=�E���
��Ch�f�
=��(��5�*C[��                                    Bxߺ�  "          @�ff�  �B�\��ff���
Ci}q�  ����C33�:�RCZxR                                    Bxߺ&6  
�          @�ff�
=�:�H������Cg��
=��z��A��9CW(�                                    Bxߺ4�  "          @�Q��o\)�@  ��{���
C?c��o\)>u���R���
C0L�                                    BxߺC�  	�          @�ff�h�ÿz�H�����
CC��h��<�����=qC3}q                                    BxߺR(  	�          @��R�e���z��\)��\)CE�3�e���G������Q�C5�3                                    Bxߺ`�  
�          @�p��,(��0�׿�����G�Ca�q�,(���33�-p��"CS�                                    Bxߺot  T          @��R��R�333��\)��=qCds3��R����?\)�6�HCS�=                                    Bxߺ~  
�          @����(��:=q�Ǯ����Cf
�(����
�1��*��CX5�                                    Bxߺ��  T          @�z�����Q녿�
=����Cl#������\�(Q��G�Cb&f                                    Bxߺ�f  "          @���
=�`  ���R��\)CtT{��
=���@���8��Cj0�                                    Bxߺ�  �          @����33�Tz���
����Cp5ÿ�33�	���<���7z�Cd��                                    Bxߺ��  �          @����I����������\CX���I�������(��G�CL                                    Bxߺ�X  �          @��\�G
=�  ���H��p�CW�f�G
=��\)�	���Q�CK�
                                    Bxߺ��  T          @�z���H�ÿ�Q���RCq�ÿ�����N�R�S33Cb�                                    Bxߺ�  �          @�(��5��4z�.{�
=C`�f�5�����(���\)CX�f                                    Bxߺ�J  �          @��:�H�7
=�^�R�?�C`}q�:�H�z��
=q����CWc�                                    Bx߻�  �          @�
=�4z��Dz����\)Cc^��4z��Q�����ffC\)                                    Bx߻�  T          @�
=�5�@�׿&ff�{Cb���5�33��\��33C[                                    Bx߻<  
�          @����N{�#�
�����u�CZu��N{��
=�\)��G�COxR                                    Bx߻-�  �          @����0���?\)���\��
=CcO\�0�׿�p��%���CW�H                                    Bx߻<�  �          @�Q��3�
�<�Ϳ�Q���{Cbk��3�
��p��\)��CW@                                     Bx߻K.  �          @����(Q��A녿������Cd�3�(Q�����,����
CX�                                     Bx߻Y�  |          @�Q��:=q�5���\����C`L��:=q���   �(�CT\)                                    Bx߻hz  ^          @�\)�9���\)���
���HC\���9����z��#�
��
CM��                                    Bx߻w   �          @�
=�C�
�333�=p��"ffC^k��C�
�z�����  CV�                                    Bx߻��  �          @��R�?\)�7��:�H�\)C_���?\)�Q��33��\)CW��                                    Bx߻�l  �          @�
=�=q�>�R������{Cg��=q�����5�,�CX�                                    Bx߻�  �          @��\)�E�����=qCj��\)��\)�<���5p�C[��                                    Bx߻��  �          @�
=�����L(��z�����CpB����ÿ�  �XQ��TC_�\                                    Bx߻�^  T          @�ff��
=�HQ��G����CnT{��
=���H�S�
�P(�C]��                                    Bx߻�  T          @�����
�B�\�(���33Co�����
��ff�Z�H�]G�C]#�                                    Bx߻ݪ  T          @��У��C33�����Cq޸�У׿��R�b�\�h{C^n                                    Bx߻�P  
�          @�p���G��I�������Cts3��G������b�\�hCb�                                     Bx߻��  �          @�
=���R�Dz��&ff��Cx����R��\)�q���Cc�)                                    Bx߼	�  �          @�ff��=q�A��,(����CzW
��=q����u�qCe�                                    Bx߼B  �          @���c�
�C�
�*�H���C}�
�c�
��=q�u��ClB�                                    Bx߼&�  �          @�p�����<���'����CuxR�����  �o\)��C^�H                                    Bx߼5�  �          @�ff�����;��1��&��Cy�׿��ÿ�33�w�Cc
=                                    Bx߼D4  �          @�
=�
=�8���@���6Q�C�>��
=��  ������Co��                                    Bx߼R�  �          @��R����:=q�?\)�5G�C��׿�Ϳ��\��G��fCqǮ                                    Bx߼a�  
�          @��0���Dz��0  �$�C���0�׿���z�H�Cq�)                                    Bx߼p&  
�          @�ff�0���Fff�.�R�"��C����0�׿����z=qQ�Cr}q                                    Bx߼~�  ,          @�ff�Q��H���*=q���Ck��Q녿���xQ���Co}q                                    Bx߼�r  �          @�{����Dz��$z��p�Cy�쿑녿�\)�qG�8RCf.                                    Bx߼�  �          @��Ϳ����K��33�Q�Cw����Ϳ�{�e��n��Ce��                                    Bx߼��  �          @��
��p��J=q���R���CqJ=��p��޸R�S33�Uz�Ca&f                                    Bx߼�d  T          @����G��9���\)�ffCr}q��G���  �g
=�u�
C[�                                    Bx߼�
  �          @�{>�  �'
=�N{�Kp�C��>�  �#�
���H �C��=                                    Bx߼ְ  �          @�����AG��4z��*C��H����Q��|����Cx0�                                    Bx߼�V  �          @�p���p��1��Dz��>\)C�1쾽p��\(������Cv��                                    Bx߼��  �          @���\�:=q�<(��3�RC�  ��\���
��  �Cs��                                   Bx߽�  �          @�p��L���6ff�9���1C~G��L�Ϳ�  �|(���CgO\                                   Bx߽H  �          @�{�Y���7��:�H�1
=C}�\�Y����G��~{33Cf�                                    Bx߽�  T          @�Q�?(���'��H���EG�C��?(�ÿ.{��G��=C�+�                                    Bx߽.�  �          @���?��H�%�C�
�7p�C��=?��H�0���|����C�E                                    Bx߽=:  T          @�G�?�p��1��AG��3G�C���?�p��aG�������C�4{                                    Bx߽K�  �          @�ff?h���5��>{�3��C��3?h�ÿp���\)��C�{                                    Bx߽Z�  �          @�33>u�.{�B�\�@G�C�z�>u�O\)�\)�C�q                                    Bx߽i,  �          @��\�u����QG��VQ�C�1�u��
=����¦k�Cp�{                                    Bx߽w�  �          @�G��\(���\�L(��R�Cy��\(���Q��z=qG�CJ�3                                    Bx߽�x  T          @�녿p����
�K��O�Cwٚ�p�׾\�z=q�HCJ.                                    Bx߽�  S          @��׾�ff�.{�8���9Q�C�Y���ff�aG��w�Q�Cs=q                                    Bx߽��  T          @�녾#�
�8���)���)�C�w
�#�
��33�p  ��C�&f                                    Bx߽�j  T          @��>L���(��J=q�PC�]q>L�Ϳ   �}p�¤ffC��                                    Bx߽�  T          @��H��33�\)�W��`Q�C��=��33�aG���G�¨G�CS�q                                    Bx߽϶  �          @�녿   ����N�R�S
=C�\�   ��(���  ¡.C\k�                                    Bx߽�\  T          @������H�.�R�:�H�9��C�����H�^�R�y��.Cp��                                    Bx߽�  �          @�G��:�H�'
=�=p��>(�C~s3�:�H�=p��xQ���Ca�f                                    Bx߽��  �          @��ÿ.{�'��<���=�\Cc׿.{�B�\�w�k�Cd�                                    Bx߾
N  �          @\)��{�%�,(��*�CrY���{�^�R�hQ�(�CT��                                    Bx߾�  �          @��׿�G��*�H�4z��2��CyW
��G��\(��r�\ffC\�{                                    Bx߾'�  T          @�  ��\)�p��:=q�<Q�Cu}q��\)�#�
�p���CQ�=                                    Bx߾6@  �          @~�R�n{�\)�;��?�Cy�\�n{�&ff�r�\�RCV��                                    Bx߾D�  "          @}p������(��>{�I(�Cr���׾�p��j�H#�CF=q                                    Bx߾S�  T          @���?:�H�G��Q��z�C���?:�H��(��h��z�C�B�                                    Bx߾b2  �          @���?h���B�\�p���HC�Z�?h�ÿ�{�j=q��C��                                    Bx߾p�  �          @��H?��C33�)���!�C�S3?녿��\�u���C�"�                                    Bx߾~  �          @���?^�R�J=q�Q���C��{?^�R��  �j=q.C��                                    Bx߾�$  �          @�G�?s33�L������G�C�9�?s33�����e��zQ�C�O\                                    Bx߾��  T          @���?�p��P  ��33��ffC�H�?�p������R�\�Z33C��                                     Bx߾�p  �          @��?�ff�L(���
=��=qC��R?�ff��G��R�\�Z33C��H                                    Bx߾�  �          @��H?Q��J�H���ffC�AH?Q녿���g���C�3                                    Bx߾ȼ  
�          @��
���8���:�H�533C��\���}p��\)=qC�{                                    Bx߾�b  "          @��
>8Q��5��?\)�:33C�Ǯ>8Q�c�
����k�C��\                                    Bx߾�  �          @����G��%�?\)�BC�&f��G��0���y��8RCmW
                                    Bx߾��  "          @�33�W
=�*=q�<(��9��C|���W
=�G��x��.C^޸                                    Bx߿T  �          @�33�(���9���3�
�-C���(�ÿ���y���Cm�q                                    Bx߿�  �          @��
��G��,(��E��A�C�j=��G��:�H�����CoE                                    Bx߿ �  �          @�(���G��,(��E��Cz�C�˅��G��8Q�����aHC���                                    Bx߿/F  
�          @��
��{�H���(Q��\)C��{��{����w�ǮC��                                    Bx߿=�  �          @����\)�E��,(��$(�C�l;�\)��G��x��ffC��                                    Bx߿L�  �          @��
��Q��C�
�.�R�&ffC�����Q쿜(��z�H�C}h�                                    Bx߿[8  �          @���   �S33����33C��Ϳ   �˅�p��33C|xR                                    Bx߿i�  �          @��
=�\)�Q��{���C��R=�\)����s�
�C�C�                                    Bx߿x�  "          @��
>�(��H���&ff�z�C��>�(������vff� C��                                    Bx߿�*  
�          @��
?#�
�G
=�'��  C��R?#�
�����vffQ�C��                                    Bx߿��  �          @�(�>�  �9���:=q�3��C�k�>�  �z�H�\)u�C�H                                    Bx߿�v  �          @�33���
�33�U��i�C�����
�#�
�z=q±� CN.                                    Bx߿�  �          @�=q�8Q��   �]p��oC�` �8Q�=��
��  ®L�C�                                    Bx߿��  �          @�z���G��.{�$G�C��������\�|(�L�C�<)                                    Bx߿�h  �          @��Ϳ���C�
�.{�$C��\��Ϳ�(��z�H\)Cu                                    Bx߿�  
�          @�z�J=q�8Q��7
=�.��C~��J=q�}p��{�G�Cg.                                    Bx߿��  �          @�z���H�-p��E��A
=C��{���H�8Q���G��=Ck�                                    Bx߿�Z            @��;Ǯ�4z��AG��:��C�3�Ǯ�Y����G���Cu��                                    Bx��   �          @���8Q��,���I���E\)C��8Q�.{���H z�CL�                                    Bx���  "          @�z�#�
�'
=�L���K{C�5þ#�
�z����H£\)C~+�                                    Bx��(L  �          @�33�k��'
=�I���H�C�q�k��������¡�
Cx��                                    Bx��6�  �          @�녾����'��A��C��C�c׾��ÿ+��|��z�Cs                                    Bx��E�  T          @��׿
=q�
=�L(��SG�C���
=q��Q��}p�¡8RCU�H                                    Bx��T>  T          @��׿fff�(Q��8Q��8{C{0��fff�B�\�u��C\33                                    Bx��b�  �          @���������<(��E  CR�R��>�
=�L(��\��C)#�                                    Bx��q�  T          @}p���33��\)�:=q�>\)C`���33���
�]p��t�HC6��                                    Bx���0  "          @~�R��=q�
=�Fff�L��Cm�Ὺ=q�B�\�o\)z�C<@                                     Bx����  T          @~{��Q��33�>�R�D  Cr����Q��
=�p  �CG^�                                    Bx���|  �          @|�Ϳ!G��+��5�7�\C�]q�!G��O\)�s�
Ch8R                                    Bx���"  
�          @|�;����5��,���-��C��q���Ϳ�G��q��fCx��                                    Bx����  T          @��þu�C33�'��"p�C�� �u���R�tz�aHC���                                    Bx���n  	�          @��\���AG��.{�(  C��ý����y��� C���                                    Bx���  �          @��?��]p���\��Q�C�E?�����b�\�t{C���                                    Bx���  �          @��\?�\�S�
�z��  C�g�?�\��\)�l��C��=                                    Bx���`  �          @��\=����<(��5��/�\C�H=��Ϳ���|(�aHC��
                                    Bx��  �          @��H=����3�
�=p��9�
C��=��Ϳ\(��\)�C�L�                                    Bx���  �          @��H=�G��4z��<(��8�RC�
=�G��aG��~�R��C�|)                                    Bx��!R  T          @��\>.{�,(��C33�A�C��
>.{�5��Q�B�C���                                    Bx��/�  �          @��\>����<(��2�\�-\)C�1�>��ÿ���z=qp�C��=                                    Bx��>�  �          @~{?aG��Z=q������G�C�33?aG����H�U��d�C�
=                                    Bx��MD  �          @|��?�\�Vff����p�C�^�?�\��ff�^{�v��C���                                    Bx��[�  �          @~�R?
=�H���ff�33C�J=?
=���H�h��Q�C��R                                    Bx��j�  �          @~{?���\�Ϳ�\)���C���?���
=�0  �B
=C��                                    Bx��y6  �          @{�>�G��w�>�G�@�Q�C�7
>�G��a녿����C��                                    Bx����  �          @|��=�\)�u�?W
=AF�RC���=�\)�l�Ϳ��
��(�C���                                    Bx����  �          @�  >�  �~{=u?L��C���>�  �Z=q�G�����C�"�                                    Bx���(  �          @�G�>�z���Q��G���G�C�R>�z��W��(���C�~�                                    Bx����  �          @�G�>k��~�R?�@���C���>k��j�H�У����C��\                                    Bx���t  �          @���>��
�\)>�(�@�33C�O\>��
�hQ��(���=qC��=                                    Bx���  �          @�G�?�\�\)�#�
�{C��H?�\�Tz��{��RC�Y�                                    Bx����  T          @�=q?5�qG���=q��ffC�W
?5�!G��Fff�G(�C�޸                                    Bx���f  �          @��H?z�H�p  ���\���\C�XR?z�H�"�\�B�\�@�C��
                                    Bx���  �          @��H?\)����=�Q�?�ffC���?\)�^�R�G����C���                                    Bx���  �          @�(�?(�����ÿz��p�C���?(���HQ��&ff�ffC��)                                    Bx��X  �          @�z�?:�H�y�������{C�Q�?:�H�.�R�@���;=qC���                                    Bx��(�  �          @��
?@  �aG����R��33C�
=?@  ���H�a��m(�C���                                    Bx��7�  �          @��
?:�H�XQ��{�  C��?:�H��(��j=q�|�C�w
                                    Bx��FJ  T          @��\?��X���
�H��C��?녿�  �g��|33C�                                      Bx��T�  �          @�G�?�R�h�ÿ�����C��
?�R���@���H��C�,�                                    Bx��c�  �          @�G�?W
=�w����
��p�C�  ?W
=�HQ��33���C���                                    Bx��r<  �          @�G�?Tz��q녿�{��C�(�?Tz��(���:�H�:G�C��                                    Bx��  �          @�G�>���\���33���C�*=>����{�c33�x�C��)                                    Bx��  T          @�G����
�p��U��a�RC��ͼ��
�\)�\)°C��\                                    Bx�.  �          @��>Ǯ�XQ������C�E>Ǯ�޸R�hQ��z�C�H�                                    Bx�¬�  �          @�=q?aG��c�
��(��ɅC��?aG��Q��U��]�C�,�                                    Bx�»z  �          @�G�?(���^�R��Q���C�e?(�ÿ����^�R�m��C�g�                                    Bx���   �          @���>��R�U��R�p�C���>��R��
=�i���C�4{                                    Bx����  �          @������HQ��!���
C��=������s33
=C�xR                                    Bx���l  �          @��þ�\)�:=q�1G��-��C�Ff��\)���
�xQ�aHC~ٚ                                    Bx���  �          @�G��#�
�H���   �(�C���#�
��{�q�(�C��
                                    Bx���  T          @��þ�33�K��=q�  C��쾳33��Q��n{p�C��                                    Bx��^  �          @�  ����S33�p���C�n��녿�33�g
=�3C�                                    Bx��"  �          @��׾k��XQ��	���z�C��k���  �fff�z�C�J=                                    Bx��0�  �          @��׼#�
�Z�H�
=���C�⏼#�
��ff�e�}�C���                                    Bx��?P  T          @���=L���aG���z���C�^�=L�Ϳ��R�^{�pp�C���                                    Bx��M�  T          @���>\�o\)��(���Q�C��>\�"�\�@  �E�
C�H�                                    Bx��\�  "          @��\?���r�\�z�H�_\)C��?���-p��4z��0
=C��{                                    Bx��kB  �          @�=q?���l�Ϳ�G���(�C���?���\)�@���?
=C�J=                                    Bx��y�  T          @��?s33�p  �������\C�q?s33�{�H���Gz�C��f                                    Bx�È�  T          @��
?����p  ��ff��
=C��3?���� ���Dz��Az�C��f                                    Bx�×4  T          @�(�?�Q��h�ÿ\����C�\?�Q��33�Mp��L
=C��                                    Bx�å�  �          @�p�?�p��l�Ϳ�(�����C�8R?�p����L(��HG�C��R                                    Bx�ô�  �          @�ff?�Q��e�У�����C���?�Q�����QG��L��C��R                                    Bx���&  T          @�?�
=�c33�����HC��3?�
=�	���R�\�O�RC��{                                    Bx����  �          @�p�?�=q�a녿��
��=qC�\)?�=q�z��W��Wp�C�k�                                    Bx���r  �          @�?����^{��33��=qC�n?��ÿ��H�\(��]�HC��                                    Bx���  �          @�p�?��
�\(���(���p�C�7
?��
��33�^�R�bC��                                    Bx����  �          @�z�?�  �^{� ������C��?�  ����a��k  C��                                    Bx��d  �          @��?�ff�Y���	������C���?�ff��\�g��q�C�k�                                    Bx��
  �          @�z�?aG��Vff�����\C�` ?aG���z��k��{C��R                                    Bx��)�  �          @��\?z�H�N{��
�	�RC�s3?z�H���
�i���~\)C�P�                                    Bx��8V  �          @�G�?����L���	�����C�>�?��ÿ����`���u�C��                                    Bx��F�  �          @�=q?�(��I���z����RC���?�(��˅�Z�H�f��C�c�                                    Bx��U�  �          @��\?�{�E��z���\)C��?�{�У��P  �S
=C�k�                                    Bx��dH  �          @��H?�z��E���(�C��?�z���
�Y���a�RC��q                                    Bx��r�  \          @�(�?���8Q��G����C��?�녿�  �\���b��C�7
                                    Bx�ā�  �          @�33@   �,����=qC�G�@   �����Z=q�aQ�C���                                    Bx�Đ:  "          @��H?�(��.�R�33�	z�C��f?�(���{�X���a=qC�AH                                    Bx�Ğ�  "          @��H?��.�R������C�?�����aG��m=qC�\                                    Bx�ĭ�  �          @��\?�\)�333��R�(�C��?�\)�����e��v
=C�:�                                    Bx�ļ,  �          @��?�\)�2�\�&ff��RC��?�\)��G��k�p�C���                                    Bx����  ~          @��\?�
=�(Q��1G��+(�C�C�?�
=�G��o\)�C��=                                    Bx���x  T          @�=q?�����R�@  �?�C�0�?��׿��w
==qC�L�                                    Bx���  \          @�=q?\(��(Q��<���;\)C�{?\(��0���y��\)C���                                    Bx����  
�          @�=q?L���'
=�@  �?
=C�� ?L�Ϳ#�
�z�HǮC��R                                    Bx��j  T          @���?8Q��*=q�=p��<33C���?8Q�5�z�H
=C�Ф                                    Bx��  ~          @���>�
=�9���0  �,��C�{>�
=���
�w
=
=C��                                    Bx��"�  �          @���>��R�+��>{�?(�C�H�>��R�8Q��|(��=C��H                                    Bx��1\  �          @���>�=q�B�\�&ff�!�C���>�=q��(��s�
��C�7
                                    Bx��@  
�          @���?8Q��H���Q��
=C�k�?8Q쿵�k�=qC�`                                     Bx��N�  �          @���?333�AG��"�\�=qC��f?333���R�p  u�C��H                                    Bx��]N  �          @���>\�0  �:=q�9�C��>\�Q��z�H(�C�XR                                    Bx��k�  "          @���?   �5�3�
�133C���?   �p���xQ�aHC��)                                    Bx��z�  �          @�  >�\)�333�5�5  C���>�\)�c�
�xQ��C���                                    Bx�ŉ@  �          @|��>�33�Y����(���\)C��>�33��{�]p��u33C�J=                                    Bx�ŗ�  
�          @z=q>aG��\(����ݮC��\>aG���p��Vff�m{C�"�                                    Bx�Ŧ�  "          @|(�>�\)�Z=q��(���C�\)>�\)��{�]p��uC�H�                                    Bx�ŵ2  "          @|��>aG��W
=��
���C��)>aG����
�aG��|�C��                                     Bx����  �          @z�H>B�\�AG��\)�  C��
>B�\��G��mp�C�`                                     Bx���~  "          @{�>����>�R�!G�� �C�1�>��ÿ�(��mp�u�C��\                                    Bx���$  "          @{�>�\)�J=q�33���C���>�\)��p��g��3C�`                                     Bx����  �          @{�>#�
�HQ����ffC�s3>#�
���j=qQ�C�/\                                    Bx���p  �          @z=q>�=q�C�
�=q�33C�z�>�=q����i���HC���                                    Bx��  �          @z�H>����@  �\)���C�)>��ÿ�  �l(�ffC�S3                                    Bx���  
�          @{�?+��8Q��$z��#��C���?+���{�l��ffC���                                    Bx��*b  "          @|��?�ff�*�H�%�$C���?�ff�h���g
=��C�u�                                    Bx��9  
�          @~{?\�4z����
=C�0�?\��
=�]p��s��C��                                    Bx��G�  "          @�  ?�33�(Q��   �=qC�3?�33�n{�`���vffC�Q�                                    Bx��VT  "          @�Q�?���+��*�H�'G�C�
=?���aG��k�z�C�f                                    Bx��d�  "          @�Q�?��
�.{�*=q�%�
C��?��
�k��l(��\C�%                                    Bx��s�  
           @|(�?�  �:�H��
�{C��=?�  ���
�`  �{�HC��                                    Bx�ƂF  
�          @z=q?Q��@  ���z�C���?Q녿�=q�c�
�\C��
                                    Bx�Ɛ�  �          @{�?xQ��?\)��C��?xQ쿨���c�
��C��                                    Bx�Ɵ�  �          @~{?Y���E�ff�C��\?Y����z��g�.C���                                    Bx�Ʈ8  �          @}p�?���C�
��R�	��C���?����Q��`���{  C��3                                    Bx�Ƽ�  
�          @�  ?�R�5��-p��,��C�"�?�R�}p��r�\�)C��3                                    Bx��˄  
�          @���?
=q�4z��333�0C�ff?
=q�p���w
=�\C��q                                    Bx���*  �          @���?��0���8Q��7  C�H�?��W
=�y���\C��{                                    Bx����  "          @�G�=�G��.{�>{�>
=C��=�G��B�\�}p���C���                                    Bx���v  
�          @���>�33�,(��>{�>\)C��q>�33�=p��|(�� C�Ǯ                                    Bx��  �          @���>���+��@���?�C�Y�>�녿5�}p���C�                                      Bx���  "          @�  ?Y���'
=�9���:�\C�?Y���333�u�C�(�                                    Bx��#h  �          @}p�?���5�!G��Q�C�?�������h��ǮC���                                    Bx��2  T          @\)?8Q��1��.�R�.�C�G�?8Q�p���q�{C���                                    Bx��@�  "          @~�R@���\)�\)�'��C�N@�    �(Q��Np�<�                                    Bx��OZ  T          @�(�@|(�>.{����  @p�@|(�?@  �h���L  A-��                                    