CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240721000000_e20240721235959_p20240722021524_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-22T02:15:24.549Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-21T00:00:00.000Z   time_coverage_end         2024-07-21T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy)h�  
z          @��H?J=q��\)�
=q��RC�Q�?J=q��  ��p����\C�K�                                    By)wf  T          @��
?J=q��  �#�
��RC�=q?J=q���׾�����C�7
                                    By)�  T          @�33?J=q��
=�0���p�C�J=?J=q��  ��\��C�B�                                    By)��  T          @��\?�R��  ����\C�#�?�R���׾�33��z�C��                                    By)�X  
�          @���?   ����   ��z�C�Y�?   ��  ���R��C�U�                                    By)��  
(          @�  >����p��&ff�G�C�=q>����{������HC�8R                                    By)��  �          @���?\)����B�\�#\)C���?\)��{����
=C�˅                                    By)�J  T          @�  ?&ff����W
=�6�RC���?&ff��z�(����RC�z�                                    By)��  �          @��?   ���   ��  C�l�?   ��ff���R��\)C�h�                                    By)�  
�          @�G�?���  ���Ϳ���C���?���  =�Q�?�
=C���                                    By)�<  �          @��>�����׾��R��p�C�+�>�����ý���\)C�(�                                    By*	�  �          @�z�?(�����\����W�C�U�?(�����H�u�Q�C�S3                                    By*�  
(          @�33?!G���G��������C�'�?!G������\)��C�%                                    By*'.  
�          @��?:�H���\�����z�C���?:�H���H�aG��3�
C��                                     By*5�  �          @��?0�����\��G�����C�� ?0����33�u�J�HC�|)                                    By*Dz  �          @���?=p�����ff�ÅC��?=p���ff����`  C��                                    By*S   �          @��?W
=��z��G����C���?W
=���;u�S33C��{                                    By*a�  
�          @�\)?k����
���
����C�N?k���(�����Q�C�J=                                    By*pl  �          @�\)?�G���33�k��C33C��?�G��������G�C��                                    By*  �          @��?(����
=��(���Q�C�w
?(������k��EC�s3                                    By*��  
Z          @�\)?W
=���
��������C���?W
=��z�L���.�RC���                                    By*�^  �          @���?G���p��   ��
=C�S3?G���{��������C�L�                                    By*�  T          @�33?333��\)�:�H��C���?333��Q������C���                                    By*��  �          @�  >�����
�k��H(�C�@ >������8Q��  C�8R                                    By*�P  �          @�\)>�
=����aG��@��C��>�
=���Ϳ.{�(�C��                                    By*��  T          @�p�>������W
=�:{C�4{>����H�#�
�p�C�.                                    By*�  T          @�(�?333�}p��h���K33C��?333��  �5��RC��q                                    By*�B  T          @�ff?
=��=q�Y���:�HC�#�?
=����&ff��C��                                    By+�  T          @�{>�(���녿fff�G�
C��>�(���33�333�=qC�H                                    By+�  	�          @�{>����=q�p���NffC��>������:�H� ��C��)                                    By+ 4  
�          @��>8Q������k��LQ�C�K�>8Q���33�5��C�H�                                    By+.�  �          @���>aG����ÿxQ��YG�C��
>aG���=q�E��*�RC��3                                    By+=�            @�(�>�z���Q�z�H�[�C�R>�z������G��,��C��                                    By+L&  �          @�ff>�\)���\�p���P(�C��{>�\)���
�:�H� ��C��\                                    By+Z�  �          @��R>L�����H�z�H�V�\C�n>L����z�B�\�'33C�h�                                    By+ir  "          @�Q�>�z����
��G��[�
C��>�z���p��L���,(�C��                                    By+x  "          @��>�z���(��n{�IC�H>�z���p��5�C��)                                    By+��  "          @��R>��H��=q�s33�Q�C�k�>��H����=p��!�C�aH                                    By+�d  
�          @��?0���z=q����o\)C���?0���}p��Y���?\)C��                                    By+�
  �          @�Q�?���xQ�\(��FffC�
=?���z�H�&ff�C�                                      By+��  T          @��H?=p��w
=��\)�~�RC�g�?=p��z�H�h���N=qC�T{                                    By+�V  �          @��
?!G��z�H����vffC���?!G��~{�aG��EG�C��=                                    By+��  T          @�(�?!G��z=q��\)�{�
C���?!G��~{�fff�JffC���                                    By+ޢ  T          @���?=p��s33����{\)C�~�?=p��w
=�aG��I�C�k�                                    By+�H  
Z          @��\?�  �qG���������C�j=?�  �u��}p��`Q�C�N                                    By+��  	�          @�p�?:�H�r�\��������C�j=?:�H�w��������C�N                                    By,
�  
�          @��?0���n�R��(�����C�E?0���tz��G���C�&f                                    By,:  
�          @��H?p���o\)��=q��{C�f?p���s�
��\)�~{C��f                                    By,'�  "          @�Q�?p���q녿fff�P��C���?p���u��0���=qC��                                    By,6�  �          @��
?��
�w��u�W�C�l�?��
�z�H�=p��$��C�U�                                    By,E,  
�          @��?�33�r�\�����r�RC�j=?�33�vff�\(��@Q�C�L�                                    By,S�  
�          @���?�p��l�Ϳ�\)���HC�.?�p��p  �h���O�C��                                    By,bx  "          @��?����o\)���\�hz�C��?����r�\�L���5C��                                    By,q  "          @�33?�
=�u��c�
�G�
C��=?�
=�w��(���z�C�q�                                    By,�  
Z          @��?�33�r�\�aG��G�C�t{?�33�u�&ff��
C�]q                                    By,�j  
�          @�G�?����qG��fff�MC�T{?����tz�+����C�:�                                    By,�  
�          @��?�{�r�\�n{�S�C�+�?�{�u�333�
=C�3                                    By,��  T          @���?���n�R�Tz��<��C��\?���q녿�����C��{                                    By,�\  	�          @���?��k��J=q�4��C���?��n{�\)��C�z�                                    By,�  
�          @x��?�\)�Vff�   ���
C���?�\)�XQ쾔z���=qC�|)                                    By,ר  
Z          @fff?��R�>�R��{��33C��?��R�?\)����(�C��{                                    By,�N  T          @c�
@G��:�H��z����C�P�@G��;���G����HC�B�                                    By,��  T          @hQ�@
=�;���33���\C��q@
=�<�;.{�,��C��=                                    By-�  
Z          @p  ?�33�K������RC�k�?�33�Mp���33��33C�O\                                    By-@  T          @tz�?ٙ��W
=�(���\)C�g�?ٙ��Y����ff�׮C�H�                                    By- �  T          @q�?��
�Q녿
=��
C�Ff?��
�S�
�\���C�*=                                    By-/�  
�          @mp�?���J=q��Q����C�T{?���K��#�
�!�C�C�                                    By->2  �          @g�?��E������
C�ff?��G
=��=q��
=C�N                                    By-L�  T          @hQ�?��E�   ��p�C�^�?��G
=�������C�C�                                    By-[~  �          @j=q@   �A녾�Q����\C���@   �C33�#�
��RC���                                    By-j$  �          @g�@��:=q����C��)@��<(���=q���C���                                    By-x�  �          @j�H@  �8Q쾏\)���
C��)@  �8�ý�Q쿴z�C��                                    By-�p  
�          @j=q@
=�=p���(���=qC���@
=�>�R�u�o\)C��H                                    By-�  �          @e�@���6ff���R��Q�C�j=@���7������HC�XR                                    By-��  	�          @b�\?����;�������33C��3?����<�ͽ��Ϳ�z�C���                                    By-�b  �          @]p�?��5���{��{C��?��6ff����   C��                                    By-�  "          @[�@Q���;k��w
=C�
=@Q��p��u����C��)                                    By-Ю  T          @\��@#33�zἣ�
����C��\@#33�z�>\)@C���                                    By-�T  
�          @`��@�
�4zᾅ�����C�\@�
�5��\)��{C�H                                    By-��  
�          @b�\@�\�7�������
C���@�\�8�þW
=�X��C��H                                    By-��  "          @g
=?�Q��@�׾����=qC�g�?�Q��A녾L���N{C�O\                                    By.F  
Z          @e�?�\)�J=q������z�C��3?�\)�K��8Q��4z�C��H                                    By.�  �          @h��?���G�����\)C���?���I���������RC��                                    By.(�  
�          @l(�?���HQ��\��(�C�y�?���J=q��\)��z�C�]q                                    By.78  "          @n{?�z��P  �0���+�C��?�z��R�\����\C�`                                     By.E�  T          @mp�?��R�S�
�B�\�<��C��?��R�Vff��� ��C���                                    By.T�  
�          @hQ�?�(��O\)�333�2�RC�,�?�(��Q녾������C�
=                                    By.c*  �          @w�?��`  �Q��D(�C�\?��c33�\)�ffC��=                                    By.q�  T          @tz�?�\)�`  �8Q��-�C��=?�\)�b�\����p�C���                                    By.�v  T          @l��?���W��=p��7�C�Ф?���Z=q���H���C��\                                    By.�  T          @fff?�z��S�
�Tz��T(�C�� ?�z��W
=�z��  C�|)                                    By.��  
�          @i��?k��\(��O\)�K�
C�~�?k��_\)����	C�c�                                    By.�h  T          @l��?Q��b�\�333�-p�C��\?Q��e���(����C�|)                                    By.�  �          @l��?@  �c�
�333�.�\C��R?@  �fff��(���C��f                                    By.ɴ  �          @l(�?5�c�
�&ff�!G�C��H?5�fff��p���=qC��3                                    By.�Z  "          @l(�?�R�e�����ffC���?�R�g��������\C��f                                    By.�   �          @dz�?:�H�\(��
=��C��R?:�H�^�R���
���C���                                    By.��  �          @fff?+��`  ����
C�e?+��a녾����(�C�Y�                                    By/L  �          @c33?�\�_\)��p���
=C��?�\�`�׽��ͿǮC�R                                    By/�  "          @b�\>�
=�_\)��p���Q�C�xR>�
=�`�׽��ͿǮC�t{                                    By/!�  �          @\(�?B�\�S�
��
=�߮C�s3?B�\�U����%�C�h�                                    By/0>  �          @Z�H?���Mp�����C�b�?���O\)�k��s�
C�N                                    By/>�  T          @_\)?��H�N�R����33C�N?��H�P�׾aG��dz�C�9�                                    By/M�  �          @W�?���HQ�+��733C�W
?���J�H��
=���HC�8R                                    By/\0  T          @W�?����AG��!G��-G�C��?����C�
�Ǯ��33C��                                    By/j�  
�          @P��?��
�*�H��ff���RC��?��
�0  �Tz��lz�C��=                                    By/y|  "          @Tz�?��H�=p����
��\)C�*=?��H�>{��\)���
C��                                    By/�"  �          @W�?�
=�8Q���� (�C�#�?�
=�:=q�k��|(�C�                                    By/��  
�          @W
=?��,(��(��'�
C��f?��.�R�Ǯ��(�C���                                    By/�n  "          @Tz�@z��#33�   ��
C��f@z��%���z�����C�XR                                    By/�  T          @U?�Q��*�H����p�C���?�Q��,�;k�����C��{                                    By/º  �          @Tz�?���*=q�!G��.�HC���?���,�;����߮C�y�                                    By/�`  T          @W
=?�ff�:=q��R�*�RC��?�ff�<�;�p���(�C��                                    By/�  T          @Vff?��
�9���(���4��C���?��
�<�;����
=C���                                    By/�  	�          @S33?�  �A녾�z���(�C�=q?�  �B�\���
=C�0�                                    By/�R  "          @`��?��H�S33���
���RC�
=?��H�R�\>W
=@Y��C�                                    By0�  T          @^{?����P      =�\)C�%?����O\)>�z�@��C�.                                    By0�  �          @n�R?���_\)=L��?G�C��?���^�R>�Q�@�
=C�+�                                    By0)D  �          @xQ�?��\�j=q=��
?�{C��3?��\�h��>���@��
C��                                     By07�  T          @tz�?�
=�g�>�Q�@��C��?�
=�dz�?+�A"�RC�#�                                    By0F�  
�          @s�
?��H�e�>�(�@�\)C�T{?��H�a�?@  A4��C�u�                                    By0U6  
�          @a�?���U>��@�
=C�˅?���R�\?5A9C��                                    By0c�  T          @X��?p���Mp�>��@��C�/\?p���J=q?@  AM��C�Q�                                    By0r�  "          @Z=q?p���O\)>��HA�
C�q?p���K�?G�AS�C�AH                                    By0�(  �          @^{?�=q�N�R?   Az�C�@ ?�=q�J�H?J=qAS�C�h�                                    By0��  �          @\��?�p��@  �B�\�Mp�C�.?�p��C33���{C��
                                    By0�t  
(          @\��?�ff�G�=��
?�z�C�J=?�ff�Fff>�p�@�G�C�\)                                    By0�  T          @h��?��C33��ff���RC�]q?��HQ�B�\�B�RC�
=                                    By0��  "          @W
=?�=q�5��fff�w�C��?�=q�:=q�!G��,��C�AH                                    By0�f  �          @S33?�p��9�����H��
C��H?�p��;��W
=�l��C�`                                     By0�  �          @S�
?����AG��\)�C��?����AG�>��@%C�                                    By0�  "          @b�\?�(��Tz�    ��C�!H?�(��S33>��R@�=qC�+�                                    By0�X  �          @g�?���XQ�<��
>�=qC�y�?���W
=>�{@�(�C��f                                    By1�  "          @e�?�ff�U�>8Q�@7
=C���?�ff�R�\?   @�\)C��H                                    By1�  �          @h��?��[�>L��@E�C�j=?��Y��?�A��C��H                                    By1"J  �          @c�
?���XQ�>\)@�C���?���Vff>��@�\C��                                    By10�  �          @]p�?�{�QG��#�
�@  C�ff?�{�P��>�=q@���C�n                                    By1?�  �          @[�?�{�N�R��G�����C�� ?�{�N�R>W
=@`  C���                                    By1N<  T          @]p�?�Q��O\)���
��=qC��?�Q��P      �#�
C�
=                                    By1\�  �          @g
=?G��^�R>�
=@�
=C�T{?G��Z�H?B�\ADQ�C�o\                                    By1k�  T          @^{?0���XQ�>Ǯ@У�C��q?0���U�?:�HAA�C���                                    By1z.  �          @[�?G��S33>��@�=qC���?G��O\)?J=qAV�RC�                                    By1��  �          @g
=?k��X��?E�AD��C���?k��S33?���A���C��                                    By1�z  �          @p��?��R�`  ?��A�C��q?��R�Z�H?h��A_\)C��\                                    By1�   T          @e?��H�U>��@�33C���?��H�QG�?O\)AQ��C�(�                                    By1��  T          @Y��?�{�L(�>Ǯ@��C���?�{�HQ�?8Q�AC�C��)                                    By1�l  "          @W�?���J=q>L��@Z�HC��\?���HQ�?�A�C��                                    By1�  �          @Y��?s33�P  =�@G�C�,�?s33�N{>�@�\)C�>�                                    By1�  �          @^{?���S33�u����C���?���R�\>�\)@�z�C�                                    By1�^  �          @\��?�33�H��>��@�G�C��?�33�E?
=Az�C�33                                    By1�  
�          @k�?���]p�<��
>�  C��?���\(�>\@�  C�&f                                    By2�  T          @r�\?����g
=����\C���?����g
=>�  @qG�C���                                    By2P  �          @c33?�33�Vff>L��@N{C�� ?�33�S�
?��A  C��)                                    By2)�  �          @\(�?�{�P  >#�
@'
=C�g�?�{�Mp�?   A�HC��                                     By28�  "          @X��?�ff�Mp�>�z�@�p�C��?�ff�J=q?!G�A,��C�4{                                    By2GB  T          @X��?�=q�J�H>���@��HC�ff?�=q�G
=?=p�AK�C��3                                    By2U�  "          @Z=q?�{�J=q?&ffA.�HC���?�{�Dz�?}p�A��\C��                                    By2d�  
(          @`��?�33�N�R?=p�AAC�˅?�33�HQ�?��A�(�C��                                    By2s4  
�          @_\)?����O\)?.{A3�
C�g�?����H��?��
A��C��=                                    By2��  
�          @`  ?���P��?(��A/\)C�=q?���J=q?��\A��
C�~�                                    By2��  
�          @aG�?���R�\?z�A�C�5�?���L��?p��Ax��C�o\                                    By2�&  
�          @_\)?���S33>�Q�@�(�C��?���O\)?8Q�A?�
C��                                    By2��  �          @_\)?���Tz�>L��@UC��?���Q�?z�A  C��=                                    By2�r  T          @c�
?�  �Z=q=��
?��
C�0�?�  �XQ�>��@�Q�C�B�                                    By2�  
(          @`��?��\�S�
>�
=@�{C��{?��\�O\)?L��AR�HC��H                                    By2پ  
�          @Vff?�  �I��>�p�@���C�Ǯ?�  �E�?:�HAJ�\C��3                                    By2�d  "          @Vff?xQ��Mp������C�g�?xQ��L(�>�{@�Q�C�s3                                    By2�
  �          @I��>L�����?��HA�z�C�l�>L���  ?�(�B�C��R                                    By3�  
�          @N�R�#�
���@=qBAffC�˅�#�
��@(��B[��C�q�                                    By3V  "          @N{�\)���@��B@G�C�R�\)����@(Q�B[
=C�˅                                    By3"�  T          @N�R=��
���@Q�B&{C��
=��
�Q�@��BA  C�
                                    By31�  T          @Mp�>.{�#�
?�\)B=qC��q>.{��@
=qB+G�C�                                    By3@H  "          @QG�=�\)�   @
=B z�C��f=�\)��R@��B;�RC�޸                                    By3N�  T          @U>��2�\?�\B   C��H>��#�
@B  C�
=                                    By3]�  T          @[�?=p��G
=?�  A�ffC��?=p��<(�?�{A��
C�                                    By3l:  "          @W�>B�\��H@B/��C�C�>B�\��@'�BK\)C���                                    By3z�  
�          @Q녾��
��z�@333BkffC�� ���
���@>�RB�L�C�:�                                    By3��  
Z          @Tz�녿Ǯ@7�Bp(�Cy���녿�Q�@B�\B�  Ct��                                    By3�,  
�          @Vff�5��33@=p�Bx=qCr�׿5���\@G
=B�33Ck{                                    By3��  
Z          @U�=p����@<(�By�CqE�=p��xQ�@E�B��
Chٚ                                    By3�x  �          @\(��u��R@#�
BCz�C��u���@4z�B_�C�g�                                    By3�  
�          @[�����(�@%�BC\)C�R��Ϳ���@5�B^�HC}�=                                    By3��  �          @Z=q��
=�z�@{B9\)C�Ф��
=���R@/\)BU�C��q                                    By3�j  �          @Y��������@(�B833C�������� ��@.{BT�C�T{                                    By3�  
�          @Z=q�B�\��H@Q�B1�
C��q�B�\�ff@*�HBN�RC�ff                                    By3��  
�          @Z=q�B�\��@{B:G�C����B�\�   @/\)BWG�C�E                                    By4\  �          @[�>��(�@=qB2\)C���>��
=@,��BO�\C��3                                    By4  
�          @Y��>B�\�@*�HBO��C���>B�\��p�@:�HBl�HC�.                                    By4*�  
(          @Y��>L����\@   B=�RC�q�>L�Ϳ���@1�B[�C�޸                                    By49N  T          @^�R>\��@#33B<��C��\>\��p�@5�BY�
C�}q                                    By4G�  T          @`��>����#33@Q�B+(�C��=>����{@,(�BH�RC�4{                                    By4V�  T          @b�\>L���)��@ffB&33C�q>L���z�@+�BD�C�j=                                    By4e@  �          @c33=����7�@z�B�C��=����%�@�B-=qC�                                      By4s�  T          @c33>\)�:=q@G�B(�C�ff>\)�'�@��B)ffC��                                    By4��  
�          @dz�=����>{?��HB�C��=����,(�@B${C��                                    By4�2  �          @c�
=#�
�:=q@�\B
=C�p�=#�
�'
=@=qB*��C�}q                                    By4��  �          @c�
>#�
�8��@z�BffC��{>#�
�%�@(�B-{C���                                    By4�~  �          @dz�>�{�;�@   BQ�C�Ff>�{�(��@�B&��C���                                    By4�$  �          @e>�p��C33?���A�33C�q�>�p��1G�@  BffC���                                    By4��  �          @g�>���L(�?�33A��C��=>���;�@z�B�
C���                                    By4�p  �          @k�>����Q�?�\)A���C�|)>����AG�@33B�\C��                                    By4�  �          @mp�>�(��S33?��A�ffC�� >�(��B�\@�BffC��                                    By4��  �          @n�R>���Tz�?У�AυC�\>���Dz�@z�B{C�e                                    By5b  
�          @o\)?
=q�P  ?�G�A��C���?
=q�>�R@(�B��C�&f                                    By5  
(          @o\)>\�S33?��HA�C�J=>\�A�@	��B��C��{                                    By5#�  �          @p  ?���S�
?�33A�33C��R?���C33@B=qC��                                    By52T  �          @n{?z�H�L(�?��A�
=C���?z�H�;�@z�B=qC�9�                                    By5@�  �          @n{?p���N�R?�=qA�
=C�q?p���>�R@G�B�C���                                    By5O�  �          @n{?fff�N�R?���A�(�C���?fff�>{@�\B\)C�q�                                    By5^F  "          @i��>����I��?��A�RC��=>����7
=@p�BC�f                                    By5l�  �          @i��?�\�H��?�\A��C���?�\�6ff@��B�C�R                                    By5{�  "          @e?z��J=q?���A�=qC�+�?z��8��@�\B
�\C��H                                    By5�8  �          @hQ�?Q��P��?�=qA�=qC�?Q��B�\?��
A�ffC��                                    By5��  �          @h��?z�H�S33?�z�A���C�H�?z�H�Fff?У�AԸRC�˅                                    By5��  �          @j=q?Q��W
=?�
=A��\C��
?Q��J=q?�z�A�p�C�E                                    By5�*  T          @g
=?E��S33?�p�A�  C��R?E��E?ٙ�A�\)C�
=                                    By5��  �          @fff?@  �S33?�(�A�ffC�o\?@  �E?�Q�A�(�C�޸                                    By5�v  �          @g�?z�H�W�?\(�A[�C�#�?z�H�Mp�?��A��RC���                                    By5�  T          @c�
?z�H�S�
?Y��A\��C�7
?z�H�I��?�=qA��C��q                                    By5��  �          @aG�?u�Q�?Tz�AZ�HC�'�?u�G�?��A�
=C���                                    By5�h  �          @^�R?p���N�R?^�RAhQ�C�"�?p���Dz�?���A�{C���                                    By6  �          @c�
?Y���P  ?�A�(�C�P�?Y���A�?�33A��HC��                                    By6�  
�          @e?\(��O\)?��
A��C�h�?\(��@��?�G�A��HC��3                                    By6+Z  �          @g�?E��S33?�G�A�\)C���?E��Dz�?�  A��C�
=                                    By6:   T          @hQ�?Tz��Tz�?�
=A��
C��?Tz��G
=?�Aۙ�C��                                     By6H�  �          @hQ�?s33�W�?n{An{C���?s33�L(�?�
=A��\C�P�                                    By6WL  �          @dz�?���Tz�?E�AFffC��\?���J=q?��\A��\C��                                    By6e�  
�          @`  ?k��Mp�?��
A���C�?k��@��?�G�A�
=C���                                    By6t�  T          @XQ�?���G
=?@  AL��C�l�?���=p�?�(�A��
C��                                     By6�>  �          @X��?�ff�G�?W
=Ad(�C�H�?�ff�<��?��A��
C��f                                    By6��  �          @U?�ff�C33?\(�Amp�C���?�ff�8Q�?���A��\C��                                    By6��  �          @XQ�?����Dz�?aG�Aq�C��H?����9��?���A�=qC�n                                    By6�0  
�          @Y��?�=q�Fff?c�
At(�C��{?�=q�:�H?�\)A�Q�C�                                      By6��  
�          @W�?��C33?L��A[�C�w
?��8��?��\A��C�H                                    By6�|  T          @Z�H?���HQ�?B�\AN�\C�f?���>{?�  A�C��f                                    By6�"  "          @Y��?����HQ�?L��AZ=qC�n?����=p�?��A�z�C���                                    By6��  �          @XQ�?xQ��G
=?h��Az=qC��?xQ��:�H?�33A�\)C�(�                                    By6�n  "          @Q�?h���?\)?}p�A�z�C�h�?h���333?��HA�33C��R                                    By7  "          @QG�?aG��>{?�ffA��C�:�?aG��0��?\AܸRC��\                                    By7�  �          @QG�?G��=p�?���A�p�C�^�?G��/\)?˅A�C��                                    By7$`  �          @QG�?+��>�R?�A�
=C�]q?+��0  ?��A�(�C��                                     By73  �          @\(�?333�K�?�\)A���C�.?333�<��?У�A�(�C���                                    By7A�  T          @\(�?.{�L��?��A�p�C���?.{�?\)?���A�\)C�k�                                    By7PR  "          @W�?333�Dz�?���A�z�C�` ?333�5�?�Q�A�z�C��                                    By7^�  �          @P  ?5�;�?��HA�p�C��3?5�,(�?�
=A�p�C�h�                                    By7m�  �          @P  ?+��;�?�(�A�
=C�n?+��,(�?�Q�A���C���                                    By7|D  T          @O\)?!G��;�?�p�A�  C��?!G��+�?ٙ�A�33C��{                                    By7��  �          @S33?(���<��?���A��HC�XR?(���+�?�ffB
=C���                                    By7��  "          @W
=?!G��C33?��
A��RC��\?!G��2�\?�\A���C�U�                                    By7�6  
�          @Vff?�R�C�
?��HA��HC��
?�R�3�
?��HA�33C�4{                                    By7��  
�          @X��?�\�E?��A��C��\?�\�4z�?�ffB �C�q                                    By7ł  
�          @^{>�G��L(�?��A��
C��f>�G��:=q?�A���C�AH                                    By7�(  T          @Z=q>��E?���A�p�C�8R>��3�
?�{B�C��H                                    By7��  T          @Z=q>��G
=?���A�=qC�,�>��5?�BG�C���                                    By7�t  �          @`  ?(��H��?�A���C��f?(��5?�Q�B��C�{                                    By8   �          @g
=?(��O\)?�(�A�ffC�W
?(��;�@ ��B\)C��H                                    By8�  �          @e?   �N�R?��RA�ffC�e?   �:�H@�B	��C�ٚ                                    By8f  �          @fff?��J=q?У�A��HC�q?��5�@	��B  C��3                                    By8,  T          @a�>�p��N{?�{A�{C�>�>�p��;�?�33B(�C��\                                    By8:�  �          @_\)>���N{?�  A�=qC�(�>���<��?�ffA�z�C���                                    By8IX  �          @c33?
=�QG�?��RA��C�'�?
=�@  ?��A�33C���                                    By8W�  �          @c33?�R�Q�?�(�A�
=C�XR?�R�@��?��A�G�C��\                                    By8f�  T          @aG�?
=�P��?�
=A�33C�
?
=�@  ?�  A��C��f                                    By8uJ  �          @S33?�\�E�?�ffA�p�C��=?�\�5?˅A�RC��                                    By8��  �          @C33>�33�5�?��A�{C�z�>�33�%�?�=qA�(�C��                                    By8��  �          @G�>\�9��?�=qA���C��q>\�*=q?˅A��C�3                                    By8�<  �          @I��>�p��;�?��A�(�C��H>�p��+�?���A��HC��{                                    By8��  �          @L��?���AG�?h��A�p�C�!H?���3�
?�Q�A�C���                                    By8��  T          @c33?E��Vff?Y��A_
=C�xR?E��H��?���A�\)C���                                    By8�.  T          @^{?E��S33?@  AHz�C��3?E��Fff?��A�Q�C��
                                    By8��  �          @\��?5�Q�?E�AN�RC�)?5�E?�{A�{C�}q                                    By8�z  
�          @Y��?=p��P  ?&ffA0(�C�b�?=p��Dz�?��RA���C��)                                    By8�   
�          @Y��?+��P  ?5A@��C��
?+��Dz�?�ffA��C�/\                                    By9�  T          @S�
?J=q�J=q?��A&ffC���?J=q�?\)?�
=A�Q�C�Y�                                    By9l  
�          @Tz�?B�\�J�H?��A'�
C��H?B�\�@  ?�Q�A�p�C��                                    By9%  �          @Vff?J=q�Mp�?
=qA��C���?J=q�C33?���A�=qC�<)                                    By93�  �          @S33?Q��I��?
=qA33C�K�?Q��?\)?�\)A�\)C���                                    By9B^  �          @N{?z�H�>{?B�\A[
=C�.?z�H�1G�?��A��
C���                                    By9Q  T          @J=q?s33�;�?+�AC�C��?s33�0  ?��HA���C���                                    By9_�  "          @J�H?aG��?\)?�A{C�'�?aG��5?�=qA�
=C���                                    By9nP  �          @C33?W
=�5�?E�Ak�C�=q?W
=�(Q�?�ffA��C��{                                    By9|�  �          @8Q�?Tz�� ��?��A�ffC�&f?Tz��\)?���B�HC�%                                    By9��  T          @:=q?aG��%?xQ�A��\C�j=?aG��ff?���A�{C�G�                                    By9�B  �          @:=q?n{�(��?O\)A��C��?n{��?��A��C�n                                    By9��  T          @=p�?W
=�,��?\(�A��RC���?W
=�\)?�{AۮC�]q                                    By9��  �          @;�?:�H�,��?fffA�(�C���?:�H�{?�33A�ffC�B�                                    By9�4  
�          @;�>��*=q?�{A�Q�C�"�>����?���BQ�C���                                    By9��  �          @>{>����+�?�Q�A��
C�B�>������?�Q�B�C�                                    By9�  "          @=p�>�ff�%?�{A��C��\>�ff�G�?�B�C��)                                    By9�&  �          @<��?z��!�?�z�A�p�C�o\?z��p�?��B�HC�XR                                    By: �  �          @;�>�z��#33?��A���C�5�>�z���R?�\)B�RC���                                    By:r  �          @<��>�{�(Q�?��RA��C��>�{�?޸RBQ�C�%                                    By:  "          @@  ?#�
�.�R?n{A�z�C��q?#�
�\)?���A�C�9�                                    By:,�  �          @AG�?���*=q>��HA
=C�)?��� ��?�G�A��
C��=                                    By:;d  �          @C33?�33�1�>��A��C�>�?�33�(Q�?�G�A���C�Ф                                    By:J
  T          @=p�?�33�*�H?�A"{C���?�33� ��?�ffA�  C�E                                    By:X�  �          @>�R?�=q�(Q�>�p�@�C�aH?�=q�   ?fffA���C��R                                    By:gV  T          @@  ?�(��%>�p�@���C��{?�(��p�?c�
A�{C�xR                                    By:u�  �          @>�R?Ǯ� ��>��
@ǮC��H?Ǯ���?Tz�A�G�C��H                                    By:��  T          @;�?�(���R>�A�C�S3?�(���?z�HA�=qC�                                      By:�H  �          @<��?����!�>�(�AC�޸?������?p��A�z�C��
                                    By:��  �          @=p�?Ǯ��R>Ǯ@��HC�#�?Ǯ�?c�
A�ffC��q                                    By:��  �          @<(�?�{�#33>��HA��C��?�{���?�G�A�=qC��3                                    By:�:  T          @8��?�(��#33?�A7
=C�˅?�(���?��A��C���                                    By:��  T          @8��?�  �%�?L��A�
=C���?�  �
=?���A��C�q�                                    By:܆  �          @=p�?p���,��?@  Aj�\C��f?p����R?��AϮC�j=                                    By:�,  �          @C33?0���8Q�?+�AMG�C���?0���+�?�G�A�Q�C�=q                                    By:��  
�          @>�R>��H�3�
?aG�A���C��3>��H�#�
?���A��HC�k�                                    By;x  T          @8Q�?��)��?}p�A�G�C��?��Q�?��
B =qC��                                    By;  �          @>�R>�ff�,��?�
=A�G�C���>�ff���?�p�B��C�K�                                    By;%�  �          @9��?��&ff?�z�A��\C���?��33?�
=B�C�ff                                    By;4j  
�          @8Q�?!G��$z�?���A�ffC���?!G���?�33B�C��                                     By;C  �          @7�?!G��!�?��HA��
C��R?!G��p�?�(�B�C��                                    By;Q�  "          @5?333���?�{A�C�4{?333��\?�B"  C��H                                    By;`\  T          @5�?#�
�=q?��A��HC�l�?#�
��
?�=qB!�C��R                                    By;o  �          @6ff?Y���ff?�{A�\C���?Y���   ?�B ��C��                                    By;}�  T          @333?s33���?�=qA��
C�n?s33��?��BQ�C�4{                                    By;�N  �          @2�\?���\)?��A�G�C�w
?����33?�G�BffC�W
                                    By;��  
�          @4z�?��(�?���A�C�f?�����?��B=qC�&f                                    By;��  �          @:=q=u�,(�?���A�ffC��f=u�Q�?�z�B�\C��)                                    By;�@  "          @<(�=�\)�,(�?�
=A�
=C�˅=�\)��?޸RB{C��                                    By;��  
�          @:�H>8Q��*�H?�A�G�C���>8Q��ff?�p�B=qC�+�                                    By;Ռ  T          @9��>����%?�p�A�G�C�]q>������?�\B�
C�H                                    By;�2  �          @7
=?
=q�{?��A��HC�#�?
=q��?���B!�C�"�                                    By;��  
�          @8Q�?#�
�{?�=qA�Q�C�9�?#�
�
=?�B�C�`                                     By<~  T          @;�?&ff�#33?��A��C��?&ff���?�=qB�C�.                                    By<$  �          @8Q�?+��{?�=qA�Q�C��3?+��
=?���B��C��=                                    By<�  T          @8��?z��   ?��A�=qC���?z��Q�?�{B G�C��R                                    By<-p  	�          @8��>��H�   ?�\)A�=qC��
>��H�Q�?�33B#�
C���                                    By<<  T          @5�?(���{?���A�p�C�l�?(�����?�p�BC��                                    By<J�  �          @5�?W
=���R?޸RBC�l�?W
=���@	��BJ�HC�>�                                    By<Yb  "          @,��?��ÿ��@BL(�C��?��ÿL��@�Bp�C���                                    By<h  T          @*�H?�  ��{@�
BJC�(�?�  �\(�@z�Bq�\C��{                                    By<v�  �          @)��?�ff��
=@�BUQ�C���?�ff�(��@�Bw�RC���                                    By<�T  
�          @*�H?u��p�@
=qBXG�C��R?u�5@��B}�\C��3                                    By<��  
�          @0  ?�  ���H@BG  C�AH?�  �p��@�Boz�C�k�                                    By<��  �          @333?Q녿���?��HB1\)C�  ?Q녿�=q@z�B`p�C��                                    By<�F  �          @2�\?aG����?���B1�C�?aG����@�
B_�C�R                                    By<��  
�          @4z�?s33��  ?�p�B3Q�C�=q?s33��  @B`ffC��=                                    By<Β  T          @3�
?\(�����?�
=B.
=C�j=?\(�����@�
B]G�C�/\                                    By<�8  
�          @1G�?k���G�?�
=B0�C�˅?k���G�@�B^�\C��                                    By<��  
�          @1�?�  ����@�B<p�C���?�  ��=q@ffBg��C�Q�                                    By<��  T          @-p�?s33����?��HB9�\C�U�?s33����@�BeC�h�                                    By=	*  T          @!G�?�\)����?���BK{C�s3?�\)�#�
@
=qBlffC�1�                                    By=�  T          @!G�?��׿�?�BG=qC���?��׿.{@��Bi��C�aH                                    By=&v  �          @!G�?�
=���?��HBLp�C�\?�
=�\)@
=qBk(�C�8R                                    By=5  
�          @!G�?Q녿�
=?��B%=qC��
?Q녿��R?��RBU  C��f                                    By=C�  �          @0��?Tz����?��B-
=C�T{?TzῨ��@��B]z�C�(�                                    By=Rh  "          @+�?Tz�޸R?�B.��C���?Tzῠ  @��B^�C��\                                    By=a  T          @0��?k���p�?�
=B2�C���?k���(�@�\B`�C�}q                                    By=o�  �          @2�\?xQ�ٙ�?��RB5�HC��=?xQ쿕@Bc��C��)                                    By=~Z  
�          @0  ?s33��  ?��B.{C�B�?s33���R@��B\�C���                                    By=�   �          @+�?n{��G�?��RBAffC���?n{�}p�@33Bm��C���                                    By=��  
�          @(Q�?k���z�@�BI33C��q?k��aG�@�
Bt�C�:�                                    By=�L  
�          @0��?}p��У�?��RB8�
C��?}p�����@�Bf�C�H                                    By=��  �          @.{?u����?�p�B;  C�j=?u����@�
Bh��C��q                                    By=ǘ  �          @0  ?p�׿У�@   B;{C��?p�׿��@BiG�C�j=                                    By=�>  
�          @+�?h�ÿ��
?��RB@z�C�g�?h�ÿ�  @33Bn33C�5�                                    By=��  �          @+�?k���z�?��B3�HC�t{?k����@\)Bc=qC�U�                                    By=�  �          @2�\?O\)��
=?�=qB$=qC�\)?O\)��
=@  BWffC�˅                                    By>0  
�          @4z�?J=q���R?�=qB"33C��?J=q��p�@G�BV{C�
=                                    By>�  T          @7
=?O\)�G�?�B �RC��?O\)��G�@�BT��C�,�                                    By>|  
�          @3�
?=p���?��B��C��R?=p��\@\)BS�\C��f                                    By>."  T          @8Q�?.{���?�ffB�HC���?.{��\)@G�BP�C�^�                                    By><�  T          @6ff?B�\�Q�?޸RB{C��q?B�\�У�@p�BKz�C���                                    By>Kn  �          @1�?E��G�?޸RB�C�k�?E��\@(�BP�C�j=                                    By>Z  �          @2�\?@  ��?�  B�
C�{?@  ���
@��BQG�C�                                      By>h�  T          @1�?L���   ?�  B\)C�޸?L�Ϳ�  @��BQ33C�f                                    By>w`  T          @4z�?L����?�p�BG�C��f?L�Ϳ���@(�BL��C�s3                                    By>�  �          @333?��\���?��
B33C�*=?��\����@��BOC�4{                                    By>��  T          @.{?�Q�޸R?�(�B
=C�  ?�Q쿠  @
=BJ�HC��                                     By>�R  
�          @-p�?B�\���H?��HB��C���?B�\���H@	��BRp�C��                                     By>��  
�          @1G�?:�H��\?�p�Bz�C��?:�H���
@(�BQ  C��\                                    By>��  T          @1G�?8Q���?޸RB��C��?8Q�\@��BRz�C���                                    By>�D  
�          @3�
?Tz���\?�  B�
C��?Tz���
@p�BO�\C�5�                                    By>��  �          @1�?E�� ��?�  B�C�~�?E����R@p�BS  C���                                    By>�  T          @0��?B�\��Q�?�B${C���?B�\��z�@  BZQ�C�q                                    By>�6  "          @0��?Tz��33?���B$�
C�� ?Tz`\)@  BZ
=C��                                    By?	�  �          @/\)?G����?�B%�C�>�?G���{@\)B[�\C���                                    By?�  �          @.�R?=p����?���B'Q�C���?=p���{@  B]�
C�\)                                    By?'(  �          @0  ?J=q���?���B&p�C�^�?J=q��{@  B\p�C�+�                                    By?5�  �          @*�H?W
=��\)?�(�B��C��?W
=��{@��BUz�C��\                                    By?Dt  �          @'�?Tz��\)?У�B�C�f?Tz῰��@�
BPG�C���                                    By?S  �          @*�H?O\)���H?�{B{C�4{?O\)��(�@z�BK�C�ff                                    By?a�  "          @*�H?Q녿�Q�?�33B=qC��f?Q녿�Q�@ffBN�
C��=                                    By?pf  
�          @,��?W
=��33?��HBQ�C��?W
=���@	��BS�\C���                                    By?  �          @0��?fff��z�?�G�Bp�C���?fff����@p�BU{C�y�                                    By?��  �          @.�R?k���p�?��B\)C�w
?k���p�@
=BIffC���                                    By?�X  �          @,(�?h�ÿ���?�{B=qC���?h�ÿ���@z�BI\)C�                                    By?��  "          @1G�?k���
?�{B(�C��?k��Ǯ@ffBD��C�L�                                    By?��  �          @1�?fff��?���B�C��H?fff��=q@ffBD\)C��H                                    By?�J  
�          @.{?k���?��RB  C��{?k�����?��RB<Q�C���                                    By?��  �          @,(�?aG��z�?�(�B\)C�s3?aG���{?�(�B<\)C�L�                                    By?�  �          @0  ?G���?�=qA�RC�q�?G���?��B-�C�z�                                    By?�<  
y          @/\)?0���Q�?�
=A�G�C��?0�׿�p�?�\B!{C��{                                    By@�  T          @-p�?!G���H?��A�ffC�N?!G��33?�33BQ�C���                                    By@�  T          @1G�?�  �
=?�p�B ��C��3?�  ���?��RB833C��H                                    By@ .  "          @1G�?�Q����?ǮB	Q�C���?�Q쿺�H@�B<��C��)                                    By@.�  �          @1�?�G�� ��?���A�
=C��?�G����?�Q�B0��C��f                                    By@=z  �          @0��?��\���H?��HB {C�n?��\���R?�Q�B2�C�.                                    By@L   "          @0��?�����H?��HA���C��=?�����R?�Q�B2{C�p�                                    By@Z�  T          @1�?�(��G�?��HA��
C��{?�(���ff?��HB333C�*=                                    By@il            @5?��R�  ?��HA�33C�u�?��R��=q?��
B�HC��                                    By@x  
y          @333?�=q�	��?���A�
=C��?�=q�޸R?�  B{C���                                    By@��  T          @2�\?����33?��A�=qC�W
?��ÿ�{?���B&�C��                                    By@�^  T          @1�?�{�  ?�
=A�{C�/\?�{��?�G�B=qC���                                    By@�  
Z          @1�?���  ?��HA��C��H?����=q?��
B��C�W
                                    By@��  
�          @.�R?�{�G�?���B(�C�ff?�{���
?���B7�C��=                                    By@�P  
�          @.�R?�G���Q�?�33B�C���?�G���33@�BK=qC���                                    By@��  
�          @/\)?s33��z�?��HB
=C�G�?s33��{@�BR�RC��                                    By@ޜ  T          @.{?s33��?��B	�C��?s33�\@�\BB��C��R                                    By@�B  �          @,(�?@  �G�?���B�C�,�?@  ���R@ffBNQ�C�U�                                    By@��  �          @,(�?J=q��(�?��BG�C��?J=q��Q�@�BQ33C�l�                                    ByA
�  �          @-p�?E����?�(�B(�C��f?E���33@ ��B?�C��\                                    ByA4  �          @0��?Tz���
?�  A�z�C�ٚ?Tz��{?���B)  C���                                    ByA'�  
�          @0  ?c�
��\?�(�A��HC���?c�
��{?�B%��C���                                    ByA6�  �          @+�?Tz��?�Q�B{C���?Tz����?�(�B=�C��
                                    ByAE&  �          @+�?@  �{?��A��C�L�?@  ��\?�{B0�C�xR                                    ByAS�  �          @0��?J=q�Q�?�z�A�=qC�33?J=q����?��
B!
=C��                                    ByAbr  �          @1�?.{�{?��A�33C���?.{�z�?��HB�
C�(�                                    ByAq  "          @0  ?\)�#33?W
=A��RC�/\?\)�p�?��
B��C��                                    ByA�  "          @.�R?��!G�?\(�A�p�C�l�?��
�H?��B
(�C�k�                                    ByA�d  	�          @.�R?G����?fffA�ffC��{?G���\?��B  C�e                                    ByA�
  �          @+�?�33�33?��A�z�C��?�33�˅?�=qB+��C���                                    ByA��  
Z          @.{?�z���?��A�Q�C��)?�z��{?���B+�C�޸                                    ByA�V  "          @-p�?�����?���A���C�C�?��׿�{?�{B-z�C��                                    ByA��  T          @-p�?�z��   ?�z�A�(�C��?�z��G�?�B5
=C��                                     ByAע  T          @.{?���p�?��HB
=C�J=?���(�?�(�B9��C�=q                                    ByA�H  �          @*=q?��\��\?��
BG�C���?��\��  ?�(�B?��C��R                                    ByA��  �          @)��?�(���33?��A��C�g�?�(���z�?��B4�
C�s3                                    ByB�  
�          @#33?����?��HA���C��?����z�?�Q�B&\)C�AH                                    ByB:  �          @  ?�녿��?��B133C�7
?�녿
=?��BY�C�P�                                    ByB �  �          @Q�?�{�(�?��HBU=qC���?�{��Q�?�Bi33C��                                    ByB/�  
�          @	��?��fff?ǮB:�C�O\?�����?�  BZ�
C��                                    ByB>,  �          @z�?������?��B��C���?����(�?ǮBD\)C�|)                                    ByBL�  
�          @ ��?����\?��\B��C�` ?��(�?�G�BC�
C�E                                    ByB[x  T          @�?�Q쿆ff?�G�B�RC�L�?�Q�!G�?�G�BA(�C��                                    ByBj  �          @��?�ff���
?��HB�C��R?�ff�^�R?��
B1ffC�!H                                    ByBx�  �          @�\?�����?��\B�\C�xR?���fff?�{B3�RC��
                                    ByB�j  �          @
=?��
��ff?��RA��C�Ǯ?��
��{?��B0��C���                                    ByB�  �          @  ?�Q쿨��?���BffC��\?�Q�Y��?��HBE��C��                                    ByB��  �          @�R?�(����?�ffB��C�3?�(��fff?��B=p�C��                                    ByB�\  �          @��?�33���?�Q�A���C�t{?�33�fff?��
B*G�C���                                    ByB�  �          @?�
=��33?�(�A��\C��{?�
=�xQ�?�=qB)�\C���                                    ByBШ  �          @��?����33?�{A�\)C�0�?�����R?ǮB#Q�C�=q                                    ByB�N  "          @G�?��Ϳ˅?fffA��C�'�?��Ϳ��R?��Bp�C���                                    ByB��  T          ?��R?����Q�>��AYC��?����p�?aG�AѮC�'�                                    ByB��  �          @Q�?�  ���R?Y��A�p�C��q?�  ��?�G�BC��H                                    ByC@  �          @�?�z��  ?��A�\)C��?�z῎{?��HB)=qC��                                    ByC�  T          @ff?�p����H?O\)A���C��q?�p���33?�(�B��C�o\                                    ByC(�  �          @
=?��R���R?O\)A�G�C��?��R��?�(�B33C�S3                                    ByC72  �          @33?��
���H?�RA�G�C��?��
���H?��
A��C�ff                                    ByCE�  
�          @
=q?�ff��ff?@  A�G�C��?�ff��  ?�Q�Bz�C��                                    ByCT~  
�          @�?�p���ff?�RA��C�"�?�p����?���A�p�C�                                    ByCc$  T          ?�p�?���G�>��HAeG�C�޸?����?n{A�C�R                                    ByCq�  
Z          ?�33?�����?�A���C�!H?����?uA�p�C���                                    ByC�p  �          ?�z�?��H��\)?
=qA��RC���?��H���?p��A�C�O\                                    ByC�  �          ?���?�=q����?�\Aq�C���?�=q����?c�
A�G�C�,�                                    ByC��  �          @33?�����
>���A4z�C��?������?E�A�  C�C�                                    ByC�b  T          @Q�?�{��=q>���A.�HC�<)?�{���?J=qA���C�W
                                    ByC�  T          @
=?��H��z�?�A�p�C�3?��H�z�H?��B%{C�                                      ByCɮ  T          @#33?���\?�{B C���?�����\?�G�B.z�C�5�                                    ByC�T  �          @$z�?��Ϳ���?��\A�=qC���?��Ϳ���?�Q�B$\)C��{                                    ByC��  �          @%�?����z�?�(�A�
=C�|)?������?�
=B"�C��                                    ByC��  �          @"�\?�G����H?���A�{C��
?�G����\?�{B�HC���                                    ByDF  T          @"�\?�{�У�?�=qA�p�C�S3?�{��(�?��
B�RC�z�                                    ByD�  �          @   ?�G���p�?}p�A�Q�C���?�G����?�p�B��C�8R                                    ByD!�  �          @!G�?У׿��?�G�A�33C�n?У׿��R?�(�B��C�Z�                                    ByD08  �          @��?��
��33?s33A��C�]q?��
���\?�B\)C�R                                    ByD>�  �          @�?У׿�ff?k�A�\)C�.?У׿�Q�?�{B�RC��\                                    ByDM�  �          @(�?��
��(�?L��A��
C�� ?��
����?�ffB �C�H                                    ByD\*  T          @�?��
�ٙ�?G�A��HC���?��
��\)?��
A���C��                                    ByDj�  �          @��?���=q?Y��A���C�<)?���p�?��B �
C���                                    ByDyv  �          @�?�33����?W
=A�\)C�4{?�33��p�?��B G�C���                                    ByD�  �          @��?��ÿ�\)?L��A�  C��?��ÿ��
?��\A��
C�j=                                    ByD��  �          @�?\��\)?Q�A�
=C��q?\���
?��BffC���                                    ByD�h  �          @�H?����
?Tz�A�
=C�� ?���Q�?��\A�z�C�E                                    ByD�  
�          @�?��H��G�?Q�A�(�C�5�?��H��
=?�G�A��C���                                    ByD´  �          @=q?����z�?G�A�G�C��=?�����?�Q�A�(�C�]q                                    ByD�Z  �          @�?�=q��=q?333A�z�C���?�=q���?��Aۙ�C�.                                    ByD�   �          @ff@Q�Tz�>�A<(�C�` @Q�!G�?5A���C��
                                    ByD�  T          @�
@z�^�R?
=qAW\)C��=@z�(��?J=qA��C�.                                    ByD�L  T          @�
@z�h��>�
=A)�C�+�@z�8Q�?0��A��C�\)                                    ByE�  �          @��@ �׿u>ǮA z�C�AH@ �׿G�?.{A��
C�^�                                    ByE�  �          @�?�p����>���A�\C�
=?�p��s33?=p�A�p�C�*=                                    ByE)>  T          @��?�׿�{>�G�A,��C��?�׿�33?\(�A��C�S3                                    ByE7�  �          @   ?����G�?^�RA�ffC���?����33?�ffA��C���                                    ByEF�  T          @   ?У׿�?aG�A�33C��?У׿�ff?�\)B\)C���                                    ByEU0  "          @{?�G���(�?k�A���C���?�G����?�B��C�>�                                    ByEc�  �          @   ?��Ϳ�(�?^�RA��HC�u�?��Ϳ��?���B�C��3                                    ByEr|  �          @#33?�Q��?k�A���C���?�Q쿥�?�z�B=qC�Z�                                    ByE�"  �          @#33?�=q��  ?uA�G�C���?�=q����?�p�BC���                                    ByE��  �          @$z�?�p�����?��A��
C�~�?�p�����?˅B�C�o\                                    ByE�n  �          @#�
?�33���?xQ�A�G�C�9�?�33��(�?��
B{C���                                    ByE�  �          @#33?�G�����?s33A�
=C��{?�G���z�?��RB�C�p�                                    ByE��  T          @"�\?��R����?p��A��C���?��R��?�p�B
=C�8R                                    ByE�`  "          @%�?�����?Y��A�ffC��\?����  ?�B�\C��                                    ByE�  T          @$z�?�
=�޸R?Y��A��
C��q?�
=��\)?�\)B �HC�ff                                    ByE�  �          @%?\��\)?h��A�(�C��q?\��(�?�(�B�C��                                    ByE�R  �          @'
=?�33�   ?h��A��HC�|)?�33�˅?�G�B�HC���                                    ByF�  
�          @'
=?�G����H?Q�A��\C�޸?�G���=q?�B�\C��                                    ByF�  �          @)��?�=q��z�?n{A���C��\?�=q��  ?�G�B
�
C�Ff                                    ByF"D  �          @(��?�G��G�?G�A���C�aH?�G���33?�33A��C�33                                    ByF0�  �          @+�?�녿��H?O\)A��
C��?�녿˅?�z�A�33C���                                    ByF?�  �          @+�?޸R���?@  A�  C�XR?޸R���?�=qA�Q�C�L�                                    ByFN6  �          @+�?�׿�G�?=p�A���C�w
?�׿�?��
A�z�C���                                    ByF\�  �          @*�H?�ff��=q?5Ax��C�G�?�ff���R?��
A�C�33                                    ByFk�  
�          @&ff?ٙ���?.{As
=C�O\?ٙ���G�?�  A�RC��                                    ByFz(  �          @(��?�(����?.{An=qC�4{?�(���ff?��\A���C��R                                    ByF��  �          @'
=?�׿�
=?&ffAhz�C��?�׿�\)?�
=A�Q�C���                                    ByF�t  
�          @p�@(����>���A�\C�H�@(��Y��?8Q�A�G�C�h�                                    ByF�  �          @%�?�׿�33?�RA^�\C�g�?�׿���?���AЏ\C�4{                                    ByF��  �          @&ff?���(�?!G�Aa�C�w
?���z�?�A��C�AH                                    ByF�f  "          @(Q�?�{��G�?z�AK�
C�S3?�{��(�?���A�z�C��=                                    ByF�  T          @*=q?�\���?&ffAb�RC�� ?�\��ff?��RAޣ�C�XR                                    ByF�  �          @*�H?���  ?(�ARffC���?�����?�z�A��
C�z�                                    ByF�X  "          @*�H@33��{?��A@(�C���@33��=q?��A�C���                                    ByF��  "          @(Q�?�  ���>�A*{C�k�?�  ��{?��A�z�C��                                    ByG�  
�          @$z�?�(���ff?#�
AfffC��?�(���p�?��HA�C��\                                    ByGJ  �          @,��?�ff��33?!G�AY�C��?�ff��=q?�p�A�{C�p�                                    ByG)�  
�          @0��?�Q��{?E�A��C�s3?�Q��=q?�(�B �C�
=                                    ByG8�  T          @/\)?�  �
�H?:�HAv�\C�O\?�  ��ff?�A�\)C��f                                    ByGG<  
�          @1G�?���Q�?!G�AS�
C��?�녿��?���A�G�C�>�                                    ByGU�  
(          @0��?��H�ff?�A/\)C��
?��H��ff?���A���C���                                    ByGd�  T          @0��?�G���?!G�AQ��C�g�?�G��ٙ�?��
A�z�C���                                    ByGs.  
(          @0  ?Ǯ���?8Q�As�C�\?Ǯ��\?�33A��C��
                                    ByG��  "          @0��?����p�?B�\A�C���?�������?��HB z�C�AH                                    ByG�z  "          @0��?��H�(�?O\)A���C���?��H���
?�G�B��C���                                    ByG�   �          @.�R?ٙ�� ��?333Ao33C�*=?ٙ���33?��A�Q�C���                                    ByG��  
�          @0��?�z���?333Al(�C�H�?�z��(�?�\)A�ffC�H                                    ByG�l  
Z          @.�R?�\)�33?G�A��RC��?�\)��?�
=A�\)C�3                                    ByG�  
�          @.�R?�=q�33?Y��A�(�C��)?�=q�У�?�  B�RC�\                                    ByGٸ  �          @/\)?˅�
=?333Al��C�z�?˅��  ?���A�(�C�&f                                    ByG�^  T          @.{?�\)��?#�
AZ�\C��
?�\)�޸R?���A癚C��f                                    ByG�  "          @.{?��R�p�?�AB{C��?��R���?��A�RC�8R                                    ByH�  T          @-p�?\�	��?�RAR�\C��
?\��?���A���C���                                    ByHP  T          @0  ?\�p�?(�AL��C�Ff?\��{?�=qA�C��R                                    ByH"�  
�          @.�R?�  ��?(��A`Q�C�9�?�  ����?�\)A�G�C��3                                    ByH1�  T          @,(�?����  ?z�AF�RC��=?��Ϳ�z�?��A�z�C��=                                    ByH@B  �          @+�?�G��G�?!G�AZ�HC�}q?�G���?���A��
C��f                                    ByHN�  �          @+�?�  ��\?��AN�HC�Ff?�  ��Q�?���A�RC�U�                                    ByH]�  �          @!�?�(���33?.{A~{C�)?�(����?���A��HC�Q�                                    ByHl4  �          @p�@�
����?Q�A��HC�K�@�
�8Q�?���AۮC�Z�                                    ByHz�  �          @!G�@ �׿�G�?Tz�A�  C��=@ �׿fff?��HA��C��                                    ByH��  "          @!G�?�Q��p�?+�Av�\C�)?�Q쿱�?�(�A��C�33                                    ByH�&  �          @�?��Ϳ��?�AJ�RC�� ?��Ϳ˅?��A��C�                                      ByH��  T          @�?k��ff>�(�A*=qC��
?k�����?���A�=qC�k�                                    ByH�r  �          @�H?���ff?\)AW\)C�e?����\?�G�A�z�C�o\                                    ByH�  �          @�H?�=q���?@  A�33C���?�=q��  ?�{B�C�Ǯ                                    ByHҾ  "          @(�?��\���H?+�A�{C�q�?��\��{?��B�\C�'�                                    ByH�d  "          @!G�?�ff��
?#�
Am�C�!H?�ff���H?���A�
=C��f                                    ByH�
  �          @"�\?�Q����?&ffAl��C�~�?�Q���
?���B�C��{                                    ByH��  "          @#33?�\)���?#�
Ag�C���?�\)��?�{Bp�C���                                    ByIV  �          @%�?�
=�{?
=AT  C���?�
=��\)?�=qA��C��                                    ByI�  "          @#�
?�33�p�?��AH��C�?�33���?��A�G�C���                                    ByI*�  T          @%�?���
?�\A9�C�C�?���  ?���A�z�C��                                     ByI9H  �          @%?�33���>���@�ffC���?�33��33?��
A�
=C�"�                                    ByIG�  �          @#33?��33>��HA0��C�N?���  ?�
=A܏\C�|)                                    ByIV�  �          @$z�?����?��AH(�C��?�녿�  ?�  A�\)C�AH                                    ByIe:  �          @%?�����>�G�Az�C�0�?����33?�
=A��
C��                                    ByIs�  �          @%�?�����>�Q�AC���?��ÿ�z�?���A˅C�XR                                    ByI��  �          @'
=?�ff�{>�G�A��C��?�ff��
=?�Q�A���C��{                                    ByI�,  T          @"�\?���
>��A{C�L�?����
?�{A�{C�H�                                    ByI��  �          @$z�?�Q��Q�>�@5�C�f?�Q����?^�RA�  C�9�                                    ByI�x  �          @%�?�Q��Q�=#�
?uC��?�Q��p�?J=qA��C��                                    ByI�  �          @%�?���p�>�  @���C�!H?����p�?�G�A��C���                                    ByI��  "          @)��?�\)��>ǮA�C�  ?�\)��33?��A�p�C��{                                    ByI�j  T          @*=q?����>��A��C�n?���z�?�z�AυC�L�                                    ByI�  �          @(Q�?Ǯ��
>�A!p�C��f?Ǯ��\?�z�A��
C���                                    ByI��  �          @'�?Ǯ�33>��AffC���?Ǯ��\?�{A�(�C��R                                    ByJ\  "          @'�?�=q�G�>��A'33C���?�=q��(�?�z�A��C�AH                                    ByJ  �          @'�?�33��(�>��A&{C��f?�33��
=?��A��C�5�                                    ByJ#�  �          @(��?�p���(�>�\)@�33C��)?�p��޸R?uA�33C�]q                                    ByJ2N  �          @+�?�
=��\>�p�@�\)C���?�
=���
?���A�\)C��R                                    ByJ@�  �          @+�?ٙ��   ?   A+�
C�8R?ٙ���Q�?�
=A��HC��)                                    ByJO�  �          @+�?��
���H>�p�@���C��?��
�ٙ�?��A���C�q                                    ByJ^@  �          @.{?�녿�
=>��R@љ�C�@ ?�녿�Q�?z�HA�ffC�                                      ByJl�  �          @0  ?�녿�
=>��A	p�C�1�?�녿�z�?���A�ffC�^�                                    ByJ{�  �          @0��?�=q��(�?��AI�C�t{?�=q�У�?�G�A�Q�C�33                                    ByJ�2  
�          @*=q?�\���H>�  @�33C��?�\�޸R?n{A��\C���                                    ByJ��  
�          @,��@녿�>.{@hQ�C�@녿��\?+�AeC�w
                                    ByJ�~  
�          @"�\@Q�Y�����6ffC�#�@Q�W
=>B�\@�  C�B�                                    ByJ�$  �          @'
=@���;u��Q�C�aH@��\)>.{@s33C�B�                                    ByJ��  
�          @'�@�������
����C���@�����=�Q�@z�C�E                                    ByJ�p  
Z          @(Q�@�R��{�#�
�`��C�^�@�R����>�{@�z�C��                                    ByJ�  �          @,(�@p��\���.{C��)@p���(�>�(�A=qC�9�                                    ByJ�  
�          @2�\@���þ8Q��hQ�C�w
@��G�>��HA"�\C���                                    ByJ�b  �          @B�\?޸R���?(�A9C��q?޸R�G�?�
=A�C�b�                                    ByK  �          @Dz�?�\)�=q>�{@���C���?�\)�Q�?�
=A��C���                                    ByK�  
(          @G
=?��H�%>B�\@a�C��
?��H�?�{A�\)C��                                    ByK+T  T          @E�?�33�,��>���@�RC���?�33�Q�?���A��C�@                                     ByK9�  T          @N{?�{�0��>\@ۅC�"�?�{�(�?���A��HC��)                                    ByKH�  T          @Tz�?����8��>�
=@�  C�E?����"�\?�
=A̸RC�ٚ                                    ByKWF  T          @P��?�33�A�>L��@eC�h�?�33�0��?��
A�{C�Z�                                    ByKe�  "          @J�H?����<��>���@��RC�8R?����(Q�?�\)A�p�C�U�                                    ByKt�  �          @N{?�33�<��?��A�C���?�33�#33?�=qA�C��                                    ByK�8  
Z          @S33?���<(�?�A (�C��H?���!�?���A�C�Y�                                    ByK��  �          @n�R?�{�W�?Tz�AO33C���?�{�5@   B��C���                                    ByK��  �          @\)?���fff?��\Alz�C��R?���>�R@  B	\)C��f                                    ByK�*  T          @�G�?�
=�n�R?��\Ah��C��H?�
=�G
=@33B
��C�]q                                    ByK��  �          @���?�(��g�?�ffAp��C�
=?�(��?\)@�\B
  C��                                    ByK�v  �          @��H?Ǯ�i��?�  Ab�HC��
?Ǯ�A�@��B�
C��)                                    ByK�  "          @��?��l(�?h��AO�
C���?��G
=@(�B��C�C�                                    ByK��  "          @�Q�?\�h��?5A$z�C�Z�?\�HQ�?��RA�C��R                                    ByK�h  
�          @y��?�{�XQ�>�@�Q�C�o\?�{�>�R?�33A�p�C���                                    ByL  T          @w
=?�Q��Q�?��A\)C�Q�?�Q��7
=?��HA�(�C�)                                    ByL�  
�          @|��?�ff�^{?\)A
=C���?�ff�A�?��A��C�S3                                    ByL$Z  "          @|��?����\(�?5A$��C���?����<(�?�z�A���C�ٚ                                    ByL3   T          @w�?��U�?L��A>ffC�K�?��3�
?��HA�\C�l�                                    ByLA�  
�          @y��@ ���QG�?(�A��C���@ ���4z�?�G�A��
C���                                    ByLPL  �          @w�@   �QG�>��H@��HC���@   �7
=?�33A�=qC��                                     ByL^�  �          @�  @�
�U�?@  A,��C��f@�
�4z�?�z�A���C��                                    ByLm�  �          @�(�@ff�Z=q?uAW�
C��{@ff�4z�@��A�(�C�U�                                    ByL|>  �          @�
=@��`��?Y��A9�C���@��=p�@z�A��HC�7
                                    ByL��  �          @�{@G��aG�?��\Ab{C���@G��9��@�RB33C�h�                                    ByL��  T          @�  ?��H�Vff?}p�Ad��C�'�?��H�0��@��B�C���                                    ByL�0  T          @z�H?����W
=?J=qA9��C�=q?����5?��HA�
=C�XR                                    ByL��  �          @p  ?�\)�Vff>�G�@�  C��q?�\)�=p�?��A�
=C�P�                                    ByL�|  "          @tz�?�=q�XQ�?O\)AC�
C���?�=q�6ff?��RA�33C��f                                    ByL�"  "          @{�@p��K�?
=A
�\C�b�@p��/\)?��HA�p�C�h�                                    ByL��  �          @�(�?�p��l(�?�{Az=qC��?�p��A�@��BQ�C���                                    ByL�n  �          @�ff?���o\)?z�HAW�C��{?���G�@�B��C�ٚ                                    ByM   �          @��?�Q��j=q?L��A2�RC�aH?�Q��G
=@�A�C�<)                                    ByM�  �          @��\?����k�?J=qA2=qC��?����H��@�A��C�N                                    ByM`  T          @��
?�p��xQ�?+�A��C���?�p��W
=@�\A�C��                                    ByM,  
Z          @�  ?��R�o\)?.{A��C�(�?��R�N�R@ ��A�33C�~�                                    ByM:�  �          @u?˅�\��?z�A(�C�Y�?˅�?\)?�A���C���                                    ByMIR  �          @~{?�z��W�?aG�AL��C���?�z��3�
@33A�\)C�
                                    ByMW�  T          @z=q@��K�?h��AW\)C��3@��'�@ ��A�Q�C�/\                                    ByMf�  T          @vff@���7
=?fffAY��C���@����?��A�=qC���                                    ByMuD  �          @|��@-p��2�\?0��A!��C�  @-p��?�A��
C���                                    ByM��  �          @w�@*=q�-p�?(��A��C�C�@*=q��?���A��HC��q                                    ByM��  �          @tz�@=q�8��?#�
A�\C��@=q���?�z�A�{C�@                                     ByM�6  �          @p��?�=q�Mp�?#�
AG�C��R?�=q�0  ?��
A�=qC��3                                    ByM��  "          @u?Ǯ�]p�?z�A
�\C�#�?Ǯ�@  ?�A�33C���                                    ByM��  
�          @s�
?�Q��fff>�{@��\C�(�?�Q��N{?��A�  C�%                                    ByM�(  �          @p��?�p��aG�>��H@�
=C���?�p��E?�  A�{C���                                    ByM��  T          @a�@333��{?333A=p�C�<)@333��(�?��A���C�/\                                    ByM�t  �          @\��@:=q��\>�A Q�C�T{@:=q��(�?���A�
=C���                                    ByM�  �          @\(�@@�׿�\)>Ǯ@�C��{@@�׿�\)?uA��C��=                                    ByN�  
�          @^{@G����?#�
A*�RC��@G���ff?�{A�
=C��                                    ByNf  �          @XQ�@=p����H?:�HAF�RC���@=p����?�(�A��\C��                                    ByN%  �          @e�@B�\��  ?5A6�HC��@B�\��{?�ffA�
=C��                                    ByN3�  �          @dz�@��1G�?�\A
=C�!H@����?�  A�(�C�4{                                    ByNBX  T          @}p�?�ff�n{>�G�@�C���?�ff�R�\?��A�(�C�Ǯ                                    ByNP�  �          @}p�?��
�s33>���@�G�C��
?��
�XQ�?�\A��C�xR                                    ByN_�  �          @}p�?���q�>�
=@�p�C��)?���W
=?��A׮C�˅                                    ByNnJ  �          @z=q?���p��>�\)@�33C��f?���X��?�33A�p�C��3                                    ByN|�  �          @w
=?@  �p��>�(�@��HC��H?@  �U�?��
Aܣ�C�S3                                    ByN��  T          @x��?���u�>Ǯ@���C�?���Z=q?��
A��C���                                    ByN�<  T          @y��?��vff>#�
@�
C�޸?��aG�?���A�ffC�:�                                    ByN��  �          @vff?�\�tz�>��@��C�Ǯ?�\�_\)?�ffA�(�C�!H                                    ByN��  �          @p  >��mp�=u?n{C���>��[�?�
=A��C��3                                    ByN�.  �          @s�
>�
=�q�>�?��HC�#�>�
=�]p�?\A�=qC�l�                                    ByN��  �          @s33>���p��>�\)@�{C��>���X��?�33A�(�C�p�                                    ByN�z  
�          @u?�\�r�\>��
@�  C���?�\�Y��?ٙ�A�{C�>�                                    ByN�   T          @\)>����|��>�p�@�Q�C���>����b�\?�ffA��HC�C�                                    ByO �  T          @g�>�p��dz�>L��@L(�C���>�p��O\)?�G�A�Q�C�Ff                                    ByOl  �          @j=q��\)�hQ�    =�\)C�k���\)�W�?���A�ffC�`                                     ByO  T          @j=q    �i����Q쿰��C�      �Z�H?��\A��HC�H                                    ByO,�  T          @mp��   �i���#�
���
C�q�   �X��?���A���C���                                    ByO;^  T          @u>L���p��>�(�@�ffC��=>L���U�?��A�G�C��)                                    ByOJ  �          @n{>��R�k�>��@��C�k�>��R�P��?�  A�z�C���                                    ByOX�  	�          @_\)��(��[��.{�333C�n��(��P  ?�{A�=qC�<)                                    ByOgP  �          @q논��qG�=�?���C��׼��]p�?�G�A��\C���                                    ByOu�  �          @k�=��j�H=u?s33C��=��X��?�A�p�C�                                    ByO��  �          @U��E��E>�G�@�33C��E��-p�?�ffA�
=C~
                                    ByO�B  �          @\(���z��B�\?8Q�AB�RCu)��z��#�
?�ffA�
=Cq�                                    ByO��  �          @X�ÿ��R�@��>B�\@N�RCs�����R�.�R?��A���CqT{                                    ByO��  �          @QG���z��A녾�=q���\Cy)��z��:�H?^�RAw�Cx\)                                    ByO�4  �          @N�R��\)�@�׾�z���p�Cy�
��\)�:=q?Y��As\)Cx��                                    ByO��  �          @5��{�'
=�aG���  Cv����{�   ?B�\Aw�
Cv\                                    ByO܀  �          @�H��R�<#�
>.{C&f��R�
�H?^�RA�G�C~�                                    ByO�&  �          @=q�+���
=�@;�C}��+��ff?xQ�A�(�C|L�                                    ByO��  �          @\)�@  �
=>.{@���C|O\�@  �Q�?��A�p�Cz�                                    ByPr  �          @�ÿ8Q��G�>#�
@vffC|}q�8Q��33?}p�A�=qCz�q                                    ByP  �          @�ͿE���
���
���C{}q�E��
=q?W
=A��HCz=q                                    ByP%�  T          @'��=p�� ��=�G�@ ��C}�Ϳ=p���\?��A��\C|\                                    ByP4d  �          @#�
��ff�33��G����Cun��ff��?B�\A��RCt33                                    ByPC
  �          @zῦff�Ǯ�W
=���
Cf&f��ff���
>\A-p�Ce��                                    ByPQ�  �          @�Ϳu�  >8Q�@�Q�Cv�R�u�G�?�  A�G�Ct��                                    ByP`V  T          @��c�
���>�@AG�Cx���c�
�33?s33A�\)Cv�{                                    ByPn�  �          @�Ϳp���  =���@�CwB��p���33?k�A�  CuW
                                    ByP}�  �          @!G������  =u?�G�CsO\�����z�?c�
A���CqW
                                    ByP�H  T          @\)���R�
=q���
�ǮCpB����R�G�?J=qA��Cnz�                                    ByP��  �          @  ��z��p�<��
>�G�Cf����z����?(��A��RCdxR                                    ByP��  �          ?�p����\��p�=���@>{Ce@ ���\���?#�
A�p�CbY�                                    ByP�:  �          ?���p���\)?=p�A�G�C^����p��E�?���B
=CT&f                                    ByP��  �          @G���{��\)?B�\A��RCe���{��(�?��BQ�C]�H                                    ByPՆ  �          @#�
��{�z�>�
=A��Cl�R��{��\?�33A���ChxR                                    ByP�,  T          @.{��=q��?���A���Cf����=q����?�p�BQ�C]&f                                    ByP��  �          @AG���=q��
?���A��Ck�3��=q�޸R?��B��Cc�\                                    ByQx  T          @@  ��\)�{>��@��Cl�ÿ�\)���?�ffA�  Ch޸                                    ByQ  �          @2�\��
=��>\)@5Cn�쿷
=�	��?�G�A�  ClL�                                    ByQ�  T          @)�������p�>�ffA(�Cn쿰�׿��?��RA���Ci�H                                    ByQ-j  �          @*�H�������>���@�=qCn�=���׿�p�?�\)Aƣ�Ck&f                                    ByQ<  �          @1녿�
=�
=>B�\@{�Cn�q��
=��?�ffA�Q�Ck�R                                    ByQJ�  �          @6ff������ͼ#�
���CoLͿ�����?c�
A�ffCmxR                                    ByQY\  �          @>�R��G��Q�>aG�@���Ciuÿ�G��Q�?��A��
Cf\)                                    ByQh  �          @B�\���H�33>�z�@�G�Ce����H��?�\)A��Ca�q                                    ByQv�  �          @?\)����33>��@���Cf}q�����\?��A��Cc
                                    ByQ�N  �          @Dz���Q�>aG�@��
Cg����Q�?��A�Cc��                                    ByQ��  �          @Fff�����\)=L��?aG�Ci\)������\?uA��HCg)                                    ByQ��  �          @C�
��=q��ͽ��
��z�CiO\��=q�33?W
=A}�Cg�)                                    ByQ�@  �          @AG���Q���\�.{�U�Ce�\��Q����?.{AR�RCd��                                    ByQ��  �          @A녿���
=�8Q��\��Cgh�������?333AV�HCf0�                                    ByQΌ  �          @Dz��p��z�L���p  Ce�Ϳ�p���R?+�AK
=Cdk�                                    ByQ�2  
�          @@�׿�=q��þ��#�
Ch����=q�G�?B�\Ah��Cg
                                    ByQ��  �          @>{��ff�!녽��Q�Cn����ff���?Q�A�{Cm)                                    ByQ�~  �          @;������!녾k����HCp.�����(�?8Q�Ab�RCo@                                     ByR	$  �          @?\)����0  �L�Ϳh��CwxR����%�?xQ�A�(�Cv
                                    ByR�  �          @/\)�}p��#33���5Cx�{�}p��Q�?h��A�\)Cw}q                                    ByR&p  �          @4zῢ�\� �׾��'
=Cs쿢�\�Q�?O\)A�
=Cq�{                                    ByR5  �          @=p��\�!G���=q��  Cn�{�\�(�?.{AT  Cn�                                    ByRC�  �          @L�Ϳ�(��*�H��33��\)CmB���(��'
=?(��A?\)Cl��                                    ByRRb  �          @L(��ٙ��*=q�����\Cmc׿ٙ��(Q�?��A.ffCm                                    ByRa  �          @QG���
=�1녾�����
=Cn�׿�
=�.�R?&ffA8��Cnp�                                    ByRo�  �          @S�
���333�
=q��\Co.���333?�A\)Co33                                    ByR~T  �          @Q녿����3�
�
=q���Cph������4z�?�A��Cpn                                    ByR��  
�          @Mp���33�,�Ϳ   �ffCn� ��33�,(�?�A��Cnp�                                    ByR��  �          @J�H����%���G����RCkG�����#�
?
=qAG�Ck�                                    ByR�F  �          @P  ����&ff����z�Cj����%?�A  Ci��                                    ByR��  �          @L�Ϳ���#33��(����Ci������!�?�A�CixR                                    ByRǒ  �          @N�R���H�!G����H���Ch.���H�!�>��A�Ch=q                                    ByR�8  �          @Fff��=q�(����*=qCi8R��=q��R>�Q�@׮Ci��                                    ByR��  �          @G����p�����2=qCi����� ��>�{@ʏ\Cj(�                                    ByR�  �          @E��ff��Ϳ
=q�!G�Ci�׿�ff�{>Ǯ@�Ci�                                    ByS*  �          @Fff��{��H�z��+�Chuÿ�{�p�>�33@θRCh�3                                    ByS�  �          @J=q��
=��ÿ5�P(�Cf�q��
=��R>W
=@z=qCh�                                    BySv  �          @Fff���H��\�0���O
=Ce}q���H���>L��@g�Cf�H                                    ByS.  �          @O\)��\)�!녿O\)�fffCi� ��\)�)��>.{@ECj                                    ByS<�  T          @P�׿�33�   �Y���r�HCh���33�(��=�@
�HCjJ=                                    BySKh  �          @L(����R�ff�Y���u�Ce����R�   =�\)?��CgxR                                    BySZ  �          @Fff���\)�k���ffCeY�����H�#�
�.{Cg�=                                    BySh�  �          @G���Q���׿n{��G�Ce\)��Q��(��#�
�J=qCg��                                    BySwZ  �          @E���	���@  �`��Cb�����=�Q�?�Cc�H                                    ByS�   �          @>�R���H��O\)�}G�Cb�H���H�  <#�
>��Cd�H                                    ByS��  �          @:�H��z���
�E��u�Cc0���z����=#�
?5Ce
=                                    ByS�L  �          @9����33��
�B�\�r�\CcLͿ�33�(�=#�
?aG�Ce)                                    ByS��  �          @;���{�Q�@  �m��Cd����{���=��
?��Cf�=                                    ByS��  �          @8Q�����z�W
=����Cd��������R���\)CfǮ                                    ByS�>  �          @:=q���
�	���B�\�s�
Cfn���
��=��
?�  Ch\                                    ByS��  �          @6ff��33�{�+��Xz�CiE��33��
>B�\@qG�CjY�                                    ByS�  �          @0  ��{���.{�f�\Ch�f��{��R>\)@4z�Cj33                                    ByS�0  �          @1녿��ff�5�l  Cg�����{=�G�@��Ci�                                    ByT	�  �          @0�׿�{�Q�.{�dz�Ch�H��{��R>\)@;�Cj(�                                    ByT|  �          @1G���\)���@  �{33Ch����\)�  =�\)?��HCj�                                    ByT'"  �          @'
=���H�33�.{�pQ�Cj�{���H�
=q=�@"�\Ck�                                    ByT5�  �          @'
=��Q����#�
�b�HCkT{��Q��
�H>#�
@`  Clu�                                    ByTDn  �          @'
=��{���#�
�dQ�CmaH��{�p�>.{@s33Cnn                                    ByTS  �          @!녿����녿&ff�n�\ClQ쿬���Q�>�@=p�Cm�=                                    ByTa�  �          @{��=q�����=p����Ck����=q��<#�
>��RCmc�                                    ByTp`  �          @
=���H��녿5��=qCmh����H�G�<��
?z�Co
                                    ByT  �          @�����{�@  ����Cm�H��� �׼��!G�CoǮ                                    ByT��  �          @ff��G������Q�����Cr����G����u��Q�Ctu�                                    ByT�R  �          @
�H��  ���ÿ&ff����Cq:῀  ��
==L��?�\)Cr�f                                    ByT��  �          @	���\(���׿
=�~�HCuO\�\(����H>�@Z=qCvG�                                    ByT��  �          @�ÿh�ÿ�\)��\�^{Ct��h�ÿ�
=>B�\@�
=Ct�R                                    ByT�D  �          @�
���\��G����R�
=qCo�ÿ��\��G�>�33A33Co�)                                    ByT��  �          @���녿�
=��
=�;�Ck�=��녿�(�>W
=@��Clff                                    ByT�  �          @�
���R���;�Q��"=qCh:῞�R��\)>u@���Ch�H                                    ByT�6  �          @ �׿�G���G�����<��Cf
��G��Ǯ>#�
@��Cf�3                                    ByU�  �          ?���(���z᾽p��4z�CeE��(�����>#�
@��Cf�                                    ByU�  �          ?�Q쿙����33����=qCe:Ῑ����G�����  Cgk�                                    ByU (  �          ?�(���  ���R��z���
Ce����  ���R>��@�{Cf                                      ByU.�  �          @�ÿ��޸R���L��Cl+������?��A�ffCj�                                     ByU=t  �          @33������
�����=qC`�)������L����ChO\                                    ByUL  
�          @\)��\)�޸R�Tz���CmQ쿏\)��
=�#�
��  Co޸                                    ByUZ�  �          @녿��H���#�
��(�ClE���H��=L��?���CmǮ                                    ByUif  T          @�H��\)����G��)�Cq�f��\)�ff>���@�Cq�H                                    ByUx  �          @�R��p�������Q��z�Ch�
��p�����>�Q�A�Ch��                                    ByU��  �          @)����R���
�!G��\��CQǮ��R�����:=qCTp�                                    ByU�X  �          @#33��{�У׿���Y��C]!H��{�޸R<��
>�C^�                                    ByU��  �          @(Q�����������'\)CVk������
=�\)?У�CW                                    ByU��  �          @�Ϳ���\)����!�CX�=����
==�G�@*�HCY�H                                    ByU�J  �          @\)�����H�Y�����Ck�f����33�8Q����Cnc�                                    ByU��  �          @zῑ녿\�E����\Ci#׿�녿ٙ��8Q�����Cl{                                    ByUޖ  �          @Q쿡G����ÿ+����
Cgff��G����H��\)���Ci�3                                    ByU�<  �          @
=q���
�޸R�=p����\CoG����
��녽�\)���HCqO\                                    ByU��  �          @�������ͿO\)����Cl��������
�B�\��33Co��                                    ByV
�  �          @
=��Q쿫��@  ��(�C^�
��Q�\�k���33CbxR                                    ByV.  �          @���H���ÿ333��(�C^
=���H��p��B�\���CaY�                                    ByV'�  �          @
=q���ÿ�{���v�RC\�����ÿ�p��L�Ϳ��HC_8R                                    ByV6z  �          @�R���
��p��#�
��(�C`{���
��\)���
���RCb��                                   ByVE   �          @�����녿E����CYxR�����þk�����C\�3                                   ByVS�  �          @3�
���G��fff��Q�CPB����  ������
CT�
                                    ByVbl  �          @4z��(����
�h�����
CV��(���  ���R����CZ��                                    ByVq  �          @:�H�\)�˅�z�H��G�CWT{�\)��=q��Q���=qC[E                                    ByV�  �          @;��  ��=q��G�����CW(��  ��=q�\���C[=q                                    ByV�^  �          @=p��G���ff��=q��p�CVk��G���=q�����CZ��                                    ByV�  �          @:�H�{��  ��z�����CU��{���\)�0z�C[
                                    ByV��  �          @7
=�G����ÿ������
CR:��G��У׿
=�?33CW�                                     ByV�P  �          @:=q�����ÿ�����\CQ}q����녿(��@z�CW
=                                    ByV��  T          @5��
=q��Q쿎{����CU� �
=q�޸R���-CZ�
                                    ByVל  �          @1�����p����
��z�CV�����  ��ff�p�C[�                                    ByV�B  �          @*�H��p���������CV���p���z��R�Y��C[�3                                    ByV��  �          @+���Q쿬�Ϳ�G����HCV�R��Q���H�5�v{C]p�                                    ByW�  �          @#�
��=q��ff��(���
=CWs3��=q��녿0���|(�C^                                    ByW4  �          @!녿�ff��p���G���ffCVT{��ff�˅�B�\��p�C]z�                                    ByW �  "          @p���{��������(�CT(���{��p���R�j�\CZxR                                    ByW/�  �          @"�\��\)��ff��\)��
=CV���\)��{����YC\�)                                    ByW>&  �          @(Q���������Q��ׅCU������У׿+��k33C[�{                                    ByWL�  �          @.�R��녿���������CY� ��녿��:�H�y�C_޸                                    ByW[r  T          @4z���ÿ��ÿ�ff��z�CS�f��ÿ�
=�B�\�yCZ@                                     ByWj  T          @<������
=��������CU0������ÿL���{�C[�                                    ByWx�  �          @*�H�33��(��������CR�3�33�Ǯ�5�u��CY8R                                    ByW�d  �          @5��	����������{CS��	�����G���{CY�                                     ByW�
  �          @/\)�33��ff���\���CTE�33��z�@  �|��CZ�H                                    ByW��  �          @0��� �׿�\)��ff���CV@ � �׿�p��@  �|z�C\                                    ByW�V  �          @%���H����(���  CR� ���H�\�@  ����CY�R                                    ByW��  �          @%�   �����G�����CP��   ��
=�Q���p�CW�H                                    ByWТ  T          @(Q��ff�s33���\��z�CL^��ff��=q�\(����CTQ�                                    ByW�H  �          @{��Q�W
=��G���
=CKLͿ�Q쿜(��fff���CT�                                    ByW��  �          @%�z�E�������RCHh��zῗ
=��  ��{CQ�                                    ByW��  �          @7
=��\�}p������(�CKs3��\��\)�^�R��Q�CR�3                                    ByX:  �          @6ff��R��G��fff���
CV!H��R��p�������CYǮ                                    ByX�  �          @<(����(���  ��33CT!H���(���G��	��CX\)                                    ByX(�  T          @7��zῬ�Ϳ�  ���
CRL��z��{����RCV�)                                    ByX7,  T          @9���=q����u��p�CP��=q���������CT�\                                    ByXE�  �          @4z���ÿ�
=�k���
=CNE��ÿ������CRǮ                                    ByXTx  �          @3�
��ͿxQ�}p����\CI���Ϳ��R����ECN�                                    ByXc  �          @5�{�Tzΐ33��{CF}q�{��z�L����z�CME                                    ByXq�  �          @4z���
�c�
�����33CI&f��
����u���CQB�                                    ByX�j  �          @:�H���Y����(���Q�CG� ����ff�������RCP��                                    ByX�  �          @9�����O\)���
���CG0������
��z����HCP�q                                    ByX��  �          @0���\)�.{������(�CD�3�\)���׿���¸RCN��                                    ByX�\  �          @#�
�ff�녿����
=CCG��ff�}p�������CMG�                                    ByX�  �          @#33�
=��ff��{� p�C?���
=�aG������љ�CJ�=                                    ByXɨ  �          @(���{�\��\)���RC=�\�{�Q녿�z���p�CHQ�                                    ByX�N  �          @����
��녿�(���G�C?:���
�L�Ϳ�G����CI
                                    ByX��  �          @-p��G��TzῘQ���33CG�q�G���
=�W
=��CO^�                                    ByX��  T          @0  �����z�H��{CU޸������ff�
=CZ5�                                    ByY@  �          @;���\��p���R�E�C`5���\��=��
?�G�Ca��                                    ByY�  �          @=p��z��G���R�@��C`5��z��
==�Q�?�G�Cau�                                    ByY!�  �          @-p���\��33��R�R�HCZ����\��G����!G�C\�\                                    ByY02  �          @0  ��
=��׿��@  C`LͿ�
=��(�=�Q�?��
Ca�=                                    ByY>�  �          @5�����녿��1p�Cd{�����>.{@`  Cd�                                    ByYM~  �          @`  ��{�S33�8Q��<(�C{^���{�K�?n{Av=qCz�                                    ByY\$  �          @c�
��  �Tz�8Q��9��Cyn��  �L(�?n{Ar�RCx�                                    ByYj�  �          @Q녿��
�E�\)�   C{�\���
�=p�?c�
A|��Cz��                                    ByYyp  �          @W��:�H�QG��#�
��G�C��3�:�H�E?���A�p�C�XR                                    ByY�  T          @XQ쿦ff�E��B�\�QG�Cw���ff�>{?Tz�Af�HCvY�                                    ByY��  �          @S�
��(��;��Ǯ�أ�Csn��(��9��?��A&�RCs&f                                    ByY�b  �          @L�Ϳ�=q�0  ���
���Cp)��=q�-p�?��A.=qCo��                                    ByY�  �          @P  ����333�aG��~{Co�)����.{?5AHz�Cn޸                                    ByY®  �          @U��G��<(�>.{@9��Cr�׿�G��.�R?�\)A��\Cq�                                    ByY�T  �          @S33��{�C33?
=qAQ�Cz  ��{�,��?�G�A���Cw��                                    ByY��  �          @P  �@  �Dz�?=p�AR�RC�!H�@  �*=q?ٙ�A���C~=q                                    ByY�  �          @.�R��  �
=>\A  Cv�q��  �ff?���A��Ct��                                    ByY�F  �          @   ��Q��녾8Q���{Cj�H��Q��p�>��HA4z�Ci�                                    ByZ�  �          @{��p���(�<��
>�(�Ci{��p���{?(��Aw�Cgp�                                    ByZ�  T          @�׿��H��(��u�\Ce�ÿ��H�ٙ�>���A=qCeu�                                    ByZ)8  �          @���녿�=q����#�C_���녿У�=�@Dz�C`�{                                    ByZ7�  T          @\)��zῬ�Ϳ�R��\)C[��zῼ(�����y��C]�
                                    ByZF�  �          @��ff���=p����HCY#׿�ff���;��
�ffC]�                                    ByZU*  �          @�
��ff��  �   �b�RCZ�{��ff�����\)�33C\�f                                    ByZc�  �          @��33������(��@Q�CX+���33���
���
�&ffCYٚ                                    ByZrv  �          @
�H��
=���
��ff�A�CYk���
=��{���
��ffC[\                                    ByZ�  �          @QG���?��ٙ�����C����?��\�{�+�\C��                                    ByZ��  �          @{��!G�?��R�  �C���!G�?��1G��3p�C�                                    ByZ�h  �          @��
�6ff?��\)�p�C{�6ff?����/\)�'\)C�f                                    ByZ�  �          @z�H�>{?�
=���ə�C�R�>{?�ff����z�Cff                                    ByZ��  �          @tz��A�?�\)�У���\)C���A�?�G��z��ffC!�                                    ByZ�Z  �          @|(��C33?�\������Q�C�
�C33?�����\��HC &f                                    ByZ�   �          @|���7�?�(�������(�Cu��7�?��
�Q���RC��                                    ByZ�  �          @x���>{?��ÿ�p���\)Cu��>{?�
=�{�(�CT{                                    ByZ�L  �          @qG��1�?�33�ٙ���C�q�1�?�G��p��ffC��                                    By[�  �          @o\)�7�?��
��=q�ȣ�C(��7�?�Q��z��\)Cu�                                    By[�  �          @p���:=q?޸R��\)���C0��:=q?�������C�f                                    By[">  �          @u��@��?�
=��Q��хCǮ�@��?������Q�C ��                                    By[0�  �          @i����?����   �  C����?�����R�+Q�C�                                    By[?�  �          @����
�H@����z�C#��
�H?���>{�@��C��                                    By[N0  �          @��H���@&ff����Q�CB����?�G��=p��9�CY�                                    By[\�  �          @����(�@1G���(���C k��(�@ ���/\)�*z�C	}q                                    By[k|  �          @��\��{@A�� �����
B�8R��{@  �6ff�1Q�C�H                                    By[z"  �          @�(��ٙ�@Dz��
=q��\)B��
�ٙ�@\)�@���;�
B�.                                    By[��  �          @xQ�u@P�׿���{BԽq�u@"�\�.�R�4�
B�\)                                    By[�n  �          @p�׿�=q@=p���\�{B�\��=q@��7
=�F�RB�3                                    By[�  �          @tzῥ�@5��{��B��H���@ ���?\)�MB�aH                                    By[��  �          @y����33@9������	=qB�Q쿳33@��?\)�H
=B���                                    By[�`  T          @|�Ϳ��
@;��	���G�B�W
���
@��<���A�B�Ǯ                                    By[�  �          @�  �\@C�
�z���(�B�Ǯ�\@G��:�H�;p�B��                                     By[�  �          @�=q��33@<����
��B�����33@
�H�7��3Q�C8R                                    By[�R  �          @��{@.�R�G��ffC)�{?�33�@  �7  Ch�                                    By[��  �          @�{�
=q@5����\)B��=�
=q@�\�<(��3
=C�3                                    By\�  �          @�G����@8������RC�=���@��7��'�HC
p�                                    By\D  �          @�=q�%�@333�33��ffC���%�@�\�3�
�!�HC�H                                    By\)�  �          @��\��@I����
��Q�B�����@��:�H�*\)C�{                                    By\8�  �          @�ff��@8Q���R�ᙚC���@���1G��$��C	xR                                    By\G6  �          @�{�  @C�
��G��Ə\B���  @���&ff���CE                                    By\U�  �          @��R���@HQ��\��=qB�33���@���(Q��z�C�                                    By\d�  �          @��R�ff@A녿�(����B����ff@��#�
���C�=                                    By\s(  T          @�{���@Z�H��G�����Bힸ���@333�{��\B���                                    By\��  �          @�p���  @c�
���\��B�\)��  @AG�����RB�G�                                    By\�t  �          @�
=�޸R@h�ÿ�Q���p�B��޸R@G���R���
B�33                                    By\�  �          @��Ǯ@vff���
��  B�  �Ǯ@N{�'����B��                                    By\��  �          @�녿�p�@c33������RB�G���p�@@  �33�	B�                                    By\�f  �          @�(����H@g
=������33B�\���H@C33���33B�G�                                    By\�  �          @���ٙ�@_\)��=q���B��ÿٙ�@<(���
�	  B�.                                    By\ٲ  �          @�{��z�@`�׿��R���B�\��z�@?\)��R� z�B�33                                    By\�X  �          @�ff��@c�
��  ����B�{��@A��  ���B�p�                                    By\��  �          @�Q��  @k����R���B��Ϳ�  @I������B�\                                    By]�  �          @�\)���H@j�H��Q����B�
=���H@J=q�{���B�                                      By]J  �          @�z���@e��{��G�B����@A��ff�
B���                                    By]"�  �          @�(���  @e��\)��p�B�=q��  @A����  B�\                                    By]1�  T          @��R���
@j�H��\)���B�aH���
@G
=�Q��
{B�\                                    By]@<  �          @�\)��(�@n{�����p�B���(�@I����H�Q�B��f                                    By]N�  �          @�Q쿸Q�@qG�������=qB��
��Q�@Mp������HB�W
                                    By]]�  �          @�����R@n{��{���
Bߣ׿��R@J�H�Q��	{B�p�                                    By]l.  �          @�\)���@mp��\��
=Bۣ׿��@Fff�!��ffB�\                                    By]z�  �          @�  ��33@p�׿�z�����B��ÿ�33@L(��(���HB�                                     By]�z  �          @�����H@i�����\��ffB�=q���H@G��G��G�B�ff                                    By]�   �          @��׿�
=@j�H��\)���B�Q��
=@G����{B�                                    By]��  �          @�녿�G�@o\)��G���  B���G�@I���!G��G�B�L�                                    By]�l  �          @��׿�p�@k���ff���HB��f��p�@E��"�\��\B�p�                                    By]�  �          @�������@j�H��ff����B�(�����@Dz��!����B�
=                                    By]Ҹ  �          @�z΅{@vff�˅��{B��f��{@N�R�'��{B�{                                    By]�^  �          @�(����@w����
���B�Q쿫�@QG��$z����B��                                    By]�  �          @�ff��(�@z=q��Q��xQ�B�z��(�@Z=q�  ���B�                                     By]��  �          @�p��У�@|�Ϳ����g�
B�Ǯ�У�@^{������B�33                                    By^P  �          @���u@y��� ����(�Bϣ׿u@K��A��)��Bգ�                                    By^�  �          @�\)��{@|�Ϳ��
��\)BӅ��{@R�\�4z��ffB�z�                                    By^*�  �          @���  @{���\���HB�ff��  @QG��3�
��\B���                                    By^9B  �          @���=q@{���  ���
B��f��=q@Q��1��p�Bأ�                                    By^G�  �          @�{���@\)�����B�Ǯ���@W��,���z�B���                                    By^V�  �          @���{@|�Ϳ���p�B�k���{@Tz��-p���HB���                                    By^e4  �          @�p��xQ�@�  �˅��
=B�Q�xQ�@Y���(���33B��                                    By^s�  �          @��s33@z�H����B�8R�s33@P  �6ff� \)Bԅ                                    By^��  �          @�녿�Q�@��ÿ�  ���B��)��Q�@XQ��333�  Bڮ                                    By^�&  T          @���@|�Ϳ�\)���B����@Vff�)���33Bڨ�                                    By^��  
�          @�p���{@�G���33��  Bң׿�{@_\)�p���\B�#�                                    By^�r  �          @��Ϳ�\)@�=q���R��
=B��
��\)@dz��z����
B��f                                    By^�  �          @�(���ff@����������B�(���ff@a������B�.                                    By^˾  �          @��
����@�=q��z��vffB�.����@e��R����B��                                    By^�d  �          @�(��h��@��ÿ������
B�ff�h��@`  �����B��                                    By^�
  �          @���
=q@�=q��ff���B�=q�
=q@Z�H�5�33B�\                                    By^��  �          @�(��#�
@�33������G�B���#�
@dz��(���B�k�                                    By_V  �          @�Q�Q�@�����\����B�녿Q�@o\)����(�B�Ǯ                                    By_�  �          @�
=�\(�@��R���R����B�(��\(�@mp������B��                                    By_#�  �          @�Q�s33@�Q쿎{�d(�B�33�s33@s33�{��{B��                                    By_2H  �          @����:�H@�녿�z���
=B�p��:�H@Y���<(�� ��B�Q�                                    By_@�  �          @�녿G�@�녿�Q���\)B�uÿG�@hQ��A���
B�33                                    By_O�  �          @�33��33@��ÿ��
�t��B�\)��33@��������\B�Ǯ                                    By_^:  T          @������@�\)���
�u�B�Ǯ����@~�R�(���G�Bخ                                    By_l�  �          @��
�h��@����Q����B�z�h��@l(��A��=qBϞ�                                    By_{�  �          @�z��@���z���p�B���@p���AG��B�(�                                    By_�,  �          @���xQ�@��\��(���33B���xQ�@����(Q���\B�#�                                    By_��  �          @��R�c�
@�
=��
=�]�B�k��c�
@�������RB�Ǯ                                    By_�x  �          @�  ���@�G��+���
=B��H���@�ff��z�����Bνq                                    By_�  �          @����ff@�
=>��?�(�B�Q��ff@��H�����L  B�B�                                    By_��  �          @�{�У�@�(������
B���У�@��\��(���G�B�L�                                    By_�j  �          @����  @�
=�����B�LͿ�  @���Q�����B�p�                                    By_�  T          @�\)��=q@�G������q�B��ÿ�=q@�G��Ǯ���\BԔ{                                    By_�  �          @�Q쿴z�@��׾�(����B��쿴z�@�  ��z���G�BָR                                    By_�\  T          @�Q����@��;.{��p�B�Q����@�ff��\)�~�\B��                                    By`  �          @�G���Q�@�=q>W
=@B�.��Q�@��R��ff�@  B��                                    By`�  �          @��ÿ�=q@��H>�ff@��RB�33��=q@����O\)��B�p�                                    By`+N  �          @�G���@�(�=#�
?�B�LͿ�@�\)��33�^�\B�.                                    By`9�  �          @�{�Tz�@��H���Ϳ��RBȅ�Tz�@�����\�}��B�Q�                                    By`H�  �          @�  ��p�@���>�=q@[�B��ÿ�p�@�\)�W
=�,��BԀ                                     By`W@  �          @�=q�c�
@�
==#�
>�Bʔ{�c�
@��H�����_�B�G�                                    By`e�  �          @��H��  @�
=>8Q�@p�B�8R��  @��
�u�BffB���                                    By`t�  �          @�p���33@���>u@:=qBЊ=��33@��h���4��B�{                                    By`�2  �          @�p���{@�Q�>�(�@�\)Bϔ{��{@�
=�:�H��B���                                    By`��  �          @��R�s33@�33    �#�
B�uÿs33@�ff��z��e�B�8R                                    By`�~  �          @��R�xQ�@�33��\)�^�RB˽q�xQ�@�{��(��qp�B̙�                                    By`�$  T          @�
=��  @�������B̏\��  @���ff��
B͏\                                    By`��  �          @����G�@�33��
=��33B̳3��G�@�33�Ǯ����B�\                                    By`�p  �          @���}p�@�  �(����HB��H�}p�@��R���H���BΏ\                                    By`�  �          @�(���@�
=��\)�\(�B�aH��@�=q��
=�l  B�ff                                    By`�  
�          @�33�p��@�  =u?L��B˳3�p��@��
��ff�S\)B�\)                                    By`�b  
Z          @����  @���L�Ϳ�B�.��  @��H��33�h  B�                                    Bya  �          @�33��  @�\)�u�E�B�G���  @��\��z��jffB�#�                                    Bya�  �          @�(���{@���#�
�.{BϮ��{@�33��{�^�HBЅ                                    Bya$T  �          @���p�@�  >��R@vffBҏ\��p�@�{�L�����B��                                    Bya2�  �          @�G��s33@��
>u@B�\B̔{�s33@�G��Tz��+�B���                                    ByaA�  "          @��H?�Q�?���o\)�n�
BC�?�Q�?0�������
A���                                    ByaPF  
(          @��?Ǯ>B�\�l(���@�
=?Ǯ�5�g�aHC��\                                    Bya^�  T          @�G�?�  �8Q��c�
�xG�C��q?�  �Ǯ�Q��Z  C��                                    Byam�  �          @�ff@
=��(�����^��C��@
=����e�;�C�>�                                    Bya|8  
�          @��@���H����affC��R@�,���tz��;�C�y�                                    Bya��  "          @��@�Ϳ��R��p��k��C��
@���   �{��F�
C���                                    Bya��  
�          @��\@  ����=q�iz�C��@  �=q�w
=�E��C��=                                    Bya�*  �          @��
@�R������=q�~��C��
@�R��33����kp�C���                                    Bya��  
�          @���@p���\)����=qC��@p�����  �t  C��                                    Bya�v  �          @���@�R�#�
��z��RC��@�R������Q��t�\C���                                    Bya�  �          @���@���\)�����~=qC��@���=q��z��pQ�C��3                                    Bya��  �          @�  @
�H�#�
��\)�{=qC��@
�H��  ����o=qC���                                    Bya�h  "          @�33@��J=q��  �q�HC��
@��ٙ��mp��V�
C���                                    Byb   �          @���@Q�333�|(��qQ�C��@Q�����j�H�W��C���                                    Byb�  �          @��H@z�(���G��w�\C�˅@z���
�r�\�_33C��H                                    BybZ  "          @�ff?�(��}p��vff�pC��=?�(������aG��Q�C�`                                     Byb,   �          @�(�@ �׿���p  �k(�C�#�@ �׿���Y���K��C�^�                                    Byb:�  
�          @��@�\�����i���`(�C���@�\���O\)�=\)C��=                                    BybIL  "          @���@ �׿�{�aG��]��C��)@ ���Q��G
=�;  C���                                    BybW�  �          @��?�(���p��c33�cffC��)?�(�� ���K��B  C�33                                    Bybf�  "          @�=q@z���w��i�C��f@z῱��i���V
=C��f                                    Bybu>  "          @�{@�H<��
��Q��k�>�@�H�c�
�z=q�bz�C���                                    Byb��  �          @�(�@�R��\)�y���ez�C�� @�R��z��n�R�W{C��                                     Byb��  �          @��H@#33���s�
�`�C��f@#33�z�H�l(��V
=C��                                    Byb�0  
(          @�(�@(�þ.{�r�\�\�\C�1�@(�ÿ�G��j=q�QQ�C�}q                                    Byb��  
�          @�
=@�>u�����k
=@��@��+��~�R�f�C�P�                                    Byb�|  �          @�ff@&ff>�\)�y���`@�@&ff����w
=�]�C���                                    Byb�"  T          @��R@p�>��
��  �h�H@�(�@p��z��~{�f33C�aH                                    Byb��  
Z          @�{@��=��������l��@�H@�ÿJ=q�|(��e�RC��R                                    Byb�n  
�          @�{?��R��z���\)=qC�Ф?��R��(���=q�p�C�<)                                    Byb�  �          @�  @33<������t�?J=q@33�aG������k
=C��H                                    Byc�  
Z          @��@(���Q����R�y�C��
@(���G����H�m��C���                                    Byc`  �          @�ff@p�=�G������w��@6ff@p��L����=q�p  C�3                                    Byc%  �          @�
=@33>\)����r��@aG�@33�B�\�����l=qC��=                                    Byc3�  	�          @��@33=u��z��s�
?�  @33�W
=�����k��C���                                    BycBR  
�          @�z�@�
���
�����v��C��@�
��G���{�k�RC�<)                                    BycP�  �          @�@�<���=q�u
=?5@��fff��
=�l(�C��R                                    Byc_�  
Z          @��@논��
���H�y{C���@녿s33��\)�n�
C��f                                    BycnD  "          @��R@�\�#�
�����y�
C���@�\�z�H��G��o\)C�y�                                    Byc|�  
�          @��R@33����z��y\)C�� @33�u�����o�C���                                    Byc��  
Z          @���@�<����H�y33?(��@녿c�
��  �p(�C�G�                                    Byc�6  T          @�z�@�<���=q�x��?0��@녿c�
��\)�p
=C�\)                                    Byc��  	�          @�G�@
�H=#�
��Q��{��?��\@
�H�\(���p��s{C�5�                                    Byc��  �          @�G�?�33�����z�\C��=?�33���������{�RC�`                                     Byc�(  
(          @�Q�?�=q����(�z�C�?�=q�����Q��~�RC�@                                     Byc��  
�          @��@
=>�ff��G��}=qA?�@
=����G��}(�C���                                    Byc�t  �          @���@(�>�����\)�yz�A	p�@(�������R�w{C��                                    Byc�  "          @���@\)>��R��
=�w=q@�\)@\)�\)��{�t��C��                                    Byd �  
�          @���@�
>��H�����33ATz�@�
������C�b�                                    Bydf  �          @�{@
=?5����|�
A�G�@
=�W
=���R.C�33                                    Byd  �          @��@33?���p��RAyG�@33��33��ff��C�5�                                    Byd,�  �          @��?��>��
���B�AG�?�׿
=��z���C�T{                                    Byd;X  �          @�G�@ ��=�Q���33ff@{@ �׿L�������|��C�.                                    BydI�  
Z          @���@�\=�����=qu�@,(�@�\�G���Q��{\)C���                                    BydX�  �          @�G�@G�>�{���\aHA�
@G��
=q���=qC���                                    BydgJ  
�          @�Q�?��R?�\��G���Ae�?��R��Q������C���                                    Bydu�  
�          @��@�?
=q�����}��Ah(�@�������=q�C�w
                                    Byd��  
�          @��?�z�?   ��(��{Amp�?�z�\����W
C�h�                                    Byd�<  �          @�(�?�
=>�����=q��A333?�
=�z�����\)C�xR                                    Byd��  T          @��\?�{?������A��?�{��p���G�C�|)                                    Byd��  T          @�=q?��?(���  �A���?�녾�\)����C�                                      Byd�.  
�          @�=q?�\)?8Q���\)�A��?�\)�.{�����=C��3                                    Byd��  �          @���?���?Q����RA��
?��ý�\)������C��)                                    Byd�z  �          @�G�?�G�?fff��ff  A��\?�G�<��
��G�Q�?&ff                                    Byd�   T          @�=q?Ǯ?c�
��
=�A��
?Ǯ<#�
�����>�                                    Byd��  T          @��H?�ff?��\��
=�HB  ?�ff>�����\� @��                                    Byel  T          @�33?�{?�{��Q�B
z�?�{>u����3A��                                    Bye  T          @��?���?z�H���=qB �\?���=�G���33L�@|(�                                    Bye%�  T          @��?���?Y�������HA㙚?��ü��
��33L�C���                                    Bye4^  �          @�p�?�z�?B�\���aHA�?�z��G���(�u�C�                                    ByeC  �          @�=q?�{?n{��{�A��?�{=�\)��G���@%                                    ByeQ�  �          @��\?�33?J=q��\)��Aͅ?�33�u��G�(�C��                                    Bye`P  T          @��?�33?s33��
=��A�ff?�33=�Q����#�@P��                                    Byen�  
�          @��H?�Q�?�����k�A�G�?�Q�>L�������3@�z�                                    Bye}�  T          @�(�?��
?�=q���k�A�Q�?��
>�  ��G��q@��H                                    Bye�B  "          @��
?�\)?�\)��33�}ffA�?�\)>������(�A                                    Bye��  "          @���?�(�?�{��33�z=qA�z�?�(�>�z���\){Az�                                    Bye��  
(          @��H?�{?��
��\)�Bff?�{>�����H��@�
=                                    Bye�4  �          @�?�p�?�  ���B=q?�p�=�Q����R��@h��                                    Bye��  "          @�=q?��H?�ff���\)B�\?��H>.{��33.@�                                    ByeՀ  �          @�33?��?����G�B�?��>#�
����  @���                                    Bye�&  �          @��H?���?h�������3B  ?���=#�
��(��?��                                    Bye��  �          @���?�p�?��\���RL�B	�?�p�>����=q�3@��R                                    Byfr  �          @��\?��?fff��
=�
A�?��=L�����k�?�\)                                    Byf  �          @��?��?����  �B�
?��>k����
Ap�                                    Byf�  �          @��H?��?�\)��Q��B�
?��>�  ��(��A&{                                    Byf-d  �          @�=q?��?�����\)��B#=q?��>�{���G�A_\)                                    Byf<
  �          @��?�ff?����ff�qB7�?�ff>��H���
(�A���                                    ByfJ�  �          @���?��H?����ff�fB@33?��H?   ����
A��H                                    ByfYV  �          @���?�Q�?�G�����qB:p�?�Q�>�
=��z���A���                                    Byfg�  �          @���?�(�?�����p�  BB�?�(�?�����H.A�\)                                    Byfv�  �          @�  ?���?�����R�=BJ�?���>������A���                                    Byf�H  �          @���?�p�?�������BAG�?�p�?����33�A\                                    Byf��  �          @�=q?�{?�{��{aHB433?�{?
=q��33u�A�\)                                    Byf��  �          @�G�?��?�z�������B9�H?��?����=q(�A��\                                    Byf�:  �          @�G�?ٙ�?�����.B��?ٙ�?����
=� A���                                    Byf��  �          @���?�Q�?�{����B\)?�Q�?���
=��A�                                    ByfΆ  �          @���?�{?������H�B   ?�{?\)��  ��A�G�                                    Byf�,  �          @�  ?\?�����G�aHB.\)?\?+���
=8RA��                                    Byf��  �          @�  ?��
?�(������RB0{?��
?333��ff�3AŮ                                    Byf�x  �          @�ff?�z�?�����
==qB{?�z�?\)��(��A��
                                    Byg	  �          @��
?��?�p���p�(�B�?��>��H����
A�z�                                    Byg�  �          @�G�?˅?�����33��B��?˅>�����p�A�Q�                                    Byg&j  �          @��?�
=?�Q����H\)B�?�
=>���\)�=Av=q                                    Byg5  �          @���?�
=?�  ��G���B33?�
=?
=q��{�A���                                    BygC�  
�          @���?�\)?��H����{\)B(
=?�\)?@  ��p�{A�G�                                    BygR\  �          @��
?���?�������x{B�?���?.{��
=�A��
                                    Byga  �          @���?�(�?�������uQ�B  ?�(�?z���ff�A�=q                                    Bygo�  �          @�?�{?������y{B�?�{?!G�����A��R                                    Byg~N  �          @�\)?�ff?�z���p��zp�B�?�ff?0�����\L�A��R                                    Byg��  �          @���?��
?�����
=�{�B��?��
?:�H��z���A��                                    Byg��  �          @��H?�?�
=�����{
=B=q?�?333��{� A�Q�                                    Byg�@  T          @��H?�33?�������z  B
=?�33?+����A���                                    Byg��  �          @��
?��H?�\)�����x�B��?��H?#�
��u�A��
                                    Bygǌ  �          @��
@ff?�Q���  �v�HA�z�@ff>���(���AN=q                                    Byg�2  �          @�ff@   ?�=q����z  B
=@   ?(���Q��3A�33                                    Byg��  �          @�\)@33?���33�v(�B{@33?333��Q�#�A���                                    Byg�~  �          @�
=?��?�{���H�vz�B"z�?��?c�
�����A�G�                                    Byh$  T          @��R?���?��H���\�up�B-
=?���?�  �����A�                                    Byh�  �          @��H?��H?������q�B<ff?��H?�33��(���B��                                    Byhp  �          @�(�?��@�����
�f�RBgff?��?Ǯ����B�BA�                                    Byh.  �          @��?�Q�@(��vff�d\)Buz�?�Q�?�=q��(��RBT{                                    Byh<�  �          @�(�?���@3�
�Z�H�>��B��q?���@G��s33�\��Bm{                                    ByhKb  �          @�?���@	���tz��bffBi�?���?Ǯ��33�~�BF                                    ByhZ  �          @���?O\)?Ǯ��z�BzQ�?O\)?k����
��BC(�                                    Byhh�  T          @��H?���?�����yG�BNp�?���?�ff��Q��3BG�                                    ByhwT  �          @���?��?�  ����}
=B<p�?��?c�
���aHBG�                                    Byh��  �          @�{?�G�?��������o(�B'G�?�G�?xQ����Q�A�z�                                    Byh��  �          @�ff?�Q�?�=q���H�qB,Q�?�Q�?xQ������)A�p�                                    Byh�F  �          @��H?�(�?�������uQ�B@��?�(�?����
=Bz�                                    Byh��  �          @��H?�(�?��H��=q�zB=q?�(�?(���ffA�z�                                    Byh��  �          @��?��?���=q�r�HB�?��?Tz���\)�AƸR                                    Byh�8  �          @�G�?ٙ�?�  ���R�v��B%33?ٙ�?aG���(��A�p�                                    Byh��  �          @��?�G�?�{��(��}33B;\)?�G�?xQ����{BQ�                                    Byh�  �          @�?�z�?�33��z��yG�B (�?�z�?L������z�A��                                    Byh�*  �          @�p�?��?��
���
�w�\B�R?��?333��Q�� A�                                      Byi	�  �          @��?�\?�(���=q�x��B
�?�\?#�
��ff�3A��                                    Byiv  �          @�(�?�\?�{��(��|�HB 33?�\?����\A�(�                                    Byi'  �          @�?�ff?�=q��z��{B�?�ff?5������A���                                    Byi5�  �          @�?���?������H�s��B�?���?@  �����A�                                      ByiDh  �          @�ff?��?��
�����xG�B�
?��?333��G���A��\                                    ByiS  �          @�p�?ٙ�?�33��{���B�R?ٙ�?z������A���                                    Byia�  �          @�p�?�?����p��|ffA��?�?�����k�A~�R                                    ByipZ  �          @���@?E����R�x��A���@>B�\�����p�@��                                    Byi   �          @�  @
=?Q���p��u��A��@
=>������}z�@��                                    Byi��  �          @���@p�?#�
���t�
A�p�@p�=��
��\)�y�R@z�                                    Byi�L  �          @�  ?�
=?�G���
=�zA�Q�?�
=>�G������AN{                                    Byi��  �          @��\?�Q�?����  �v�HB)��?�Q�?xQ����ffA��                                    Byi��  �          @�G�?�\?�{���vffB)Q�?�\?��\��33�A�                                    Byi�>  �          @���?��H?Ǯ����z��B)ff?��H?u����L�A��H                                    Byi��  �          @���?��?����z��tp�B)��?��?�ff��=q{A��
                                    Byi�  �          @��H?�?�{��
=�v(�B&Q�?�?��\��z��3A���                                    Byi�0  �          @�33?��H?˅�����z��B+�?��H?�  ��{=qA��H                                    Byj�  �          @�33?��?�{�����|�B2�?��?��\��
=p�A�p�                                    Byj|  �          @��H?���?˅��=q��HB5�H?���?�  ���#�BQ�                                    Byj "  �          @�=q?�G�?�����=q��B:(�?�G�?�G����  B                                    Byj.�  �          @��
?���?�����33#�B6=q?���?�G�����G�B(�                                    Byj=n  �          @�{?�{?����#�B.\)?�{?p�����HǮA�{                                    ByjL  �          @�p�?�=q?����� B133?�=q?s33���\B�A�Q�                                    ByjZ�  �          @��
?�(�?�����z�W
B;=q?�(�?z�H�����=B
=                                    Byji`  �          @��?��?У���z��=BF{?��?�ff���=qB��                                    Byjx  �          @��?�{?�ff��p�Q�BB�?�{?xQ����\ǮB                                      Byj��  �          @��
?�G�?�  ��\)  BH{?�G�?k���(��{B\)                                    Byj�R  T          @�(�?�?�����k�BC{?�?@  ��{�=B=q                                    Byj��  �          @��?�G�?��R���H��B2�?�G�?&ff���R��A���                                    Byj��  �          @�{?�  ?����33�\B<\)?�  ?B�\��
=L�A���                                    Byj�D  �          @��\?���?��\����ǮB;  ?���?333��(�W
A�{                                    Byj��  
�          @��?��?����
=��B:�H?��?\(���33aHB��                                    Byjސ  �          @�(�?�(�?����G�B;��?�(�?B�\�����B (�                                    Byj�6  �          @�z�?�z�?������ffBE�
?�z�?O\)��{z�B=q                                    Byj��  �          @�z�?��\?�
=��z���BD\)?��\?!G�����=A���                                    Byk
�  �          @�Q�?��?�����
��B/(�?��?G������A�Q�                                    Byk(  �          @�G�?��\?������ffB   ?��\?�����\aHA��                                    Byk'�  �          @���?p��?�����=qaHBH\)?p��?����p��B�                                    Byk6t  �          @���?z�H?�  ����ffBP�?z�H?:�H��z��{B�                                    BykE  �          @�z�?:�H?aG�����)BIQ�?:�H>�{��� ��A�z�                                    BykS�  �          @��>�(�?xQ���G��B�>�(�>�
=���
¥�)B2Q�                                    Bykbf  �          @���?��?������aHB��)?��?&ff��p� L�BHff                                    Bykq  �          @��H?��
?�(����r�\BP?��
?�(����
B/Q�                                    Byk�  �          @��\?˅?�����sQ�BG��?˅?�33����RB%
=                                    Byk�X  
�          @���?���?�(�����o�HBM�?���?�p�����33B-33                                    Byk��  T          @���?�@\)��z��\{BJz�?�?��
����nz�B0
=                                    Byk��  �          @��H?�z�@p��w��S�BD��?�z�?��
���H�e��B,
=                                    Byk�J  �          @��?��
?�{�p���^G�B9  ?��
?����|(��o
=BG�                                    Byk��  �          @��?��?���w��`�\B9��?��?��H�����qQ�B�H                                    Bykז  �          @�Q�?�\)?�33�g
=�V=qB5��?�\)?�G��r�\�f�Bz�                                    Byk�<  �          @�33?˅@G��s�
�`
=BN�
?˅?�����  �r=qB4                                    Byk��  �          @��?��
?�  ��ff�t  BC=q?��
?�ff���z�B!=q                                    Byl�  �          @��?�G�@ ���\)�g�BTQ�?�G�?�=q���yz�B9Q�                                    Byl.  �          @�ff?˅?�(����H�vz�B<�
?˅?�G���  \)Bz�                                    Byl �  �          @��?��?˅��\)�~��B7p�?��?�\)���
G�B�                                    Byl/z  �          @��R?Ǯ?��\�����B��?Ǯ?L����z���A���                                    Byl>   �          @��\?��R?�ff�����B$  ?��R?W
=����ffA�G�                                    BylL�  T          @�ff?�ff?������z�B5=q?�ff?^�R��p��BQ�                                    Byl[l  �          @���?�
=?h�������B��?�
=>���\)k�A��H                                    Bylj  �          @��?�{?W
=���
��A�
=?�{>�����{�=A�                                    Bylx�  �          @�p�?�Q�?333��(���AΣ�?�Q�>����aHA"ff                                    Byl�^  �          @�{?�{?Q�����L�A���?�{>\���R��Av=q                                    Byl�  �          @�?��?h�����
B�B=q?��>���{G�A�=q                                    Byl��  �          @��?��?B�\���u�B
�?��>��
��
=\)A��H                                    Byl�P  �          @��
?���?E���p��3B�
?���>�{��
=A��
                                    Byl��  �          @���?�p�?c�
��z��fBQ�?�p�>����R.A��                                    BylМ  �          @��?�Q�?�ff��=q� B%�?�Q�?!G�������Aߙ�                                    Byl�B  �          @��
?�{?��������=B�R?�{?0������A��
                                    Byl��  �          @�z�?�ff?�z���G��HB'p�?�ff?@  ��z�Q�A��                                    Byl��  �          @���?��H?��\���

=B �
?��H?(���{\A�
=                                    Bym4  �          @�\)?�(�?(��������A��H?�(�>k����u�A,��                                    Bym�  �          @�G�?���?O\)����RB  ?���>\���
p�A�
=                                    Bym(�  �          @�G�?�  ?u������B��?�  ?����H\A���                                    Bym7&  �          @��H?���?��H��p���B?
=?���?����G�33B{                                    BymE�  �          @���?��
?��H��  ffBB�?��
?�����
�RB\)                                    BymTr  �          @���?���?��\��=q#�B;
=?���?Y����p�
=B�R                                    Bymc  �          @��H?�\)?����=q�=B633?�\)?8Q�����(�B=q                                    Bymq�  �          @�=q?���?�=q��=q33B5Q�?���?+������3A�\)                                    Bym�d  �          @�{?�ff?�ff��z�aHBK�R?�ff?aG�����RB33                                    Bym�
  �          @���?�G�?����{�BB\)?�G�?=p�����
=B�R                                    Bym��  �          @y��?c�
��ff�e�u�C�.?c�
��{�\���w��C�q�                                    Bym�V  T          @��\?5�\)�|(�.C��?5�h���xQ���C��                                    Bym��  �          @��\?Q�    �\)�)=L��?Q녾�33�~�R�
C�z�                                    Bymɢ  �          @�G�?u?W
=��338RB$��?u>������qA���                                    Bym�H  �          @��\?fff?������B\G�?fff?c�
����B2�                                    Bym��  �          @�{>��H?�=q���
��B�ff>��H?�������)B�aH                                    Bym��  �          @�Q�>�{?�\)���
{B�� >�{?��R��Q���B�p�                                    Byn:  �          @�33>��@ff���
��B��)>��?ٙ�����ffB�=q                                    Byn�  �          @���>�Q�@{��(�.B��\>�Q�?�=q�����B���                                    Byn!�  �          @�z�>���@Q���G��x�B���>���?��R��\)�
B��
                                    Byn0,  �          @���>�G�@=q���H�lQ�B�\)>�G�@�
�����
=B���                                    Byn>�  �          @��\?�\@=p��C33�5��B�z�?�\@,���R�\�H��B���                                    BynMx  �          @q�?=p�@E�z��z�B�33?=p�@:=q�z���RB���                                    Byn\  �          @^{?Y��@
=q�!G��=��B�Ǯ?Y��?�Q��,(��N�B��=                                    Bynj�  
�          @qG�?��@\)�:�H�N�B��?��@   �E�`  B���                                    Bynyj  �          @tz��@%��4z��>33B�=q��@�AG��P��B�aH                                    Byn�  �          @r�\�u@"�\�333�>�B��u@33�?\)�P�B�
=                                    Byn��  �          @vff��@!��8Q��B��B�#׼�@33�E��U
=B�B�                                    Byn�\  �          @|(�>B�\@��J�H�V=qB��R>B�\@��Vff�hG�B��\                                    Byn�  �          @|��>u@ff�J=q�TB�W
>u@ff�U��f�B��                                    Byn¨  �          @~{�8Q�@!G��C�
�I�HB�\)�8Q�@G��P  �[�RB�G�                                    Byn�N  �          @z=q�&ff@'
=�5�9�B���&ff@Q��A��K�BҨ�                                    Byn��  �          @|(����@:=q��\���B��H���@.{� ���
=B�q                                    Byn�  �          @�\)�333@@���5�*\)B�LͿ333@1��C�
�;p�B�W
                                    Byn�@  �          @�G����
@$z��Z=q�S
=B�=q���
@33�e�dQ�B��)                                    Byo�  �          @�  �^�R@,(��J�H�@��B��H�^�R@���W��QffB�=q                                    Byo�  T          @�녿u@1��J�H�<G�B���u@"�\�W��Lz�B�z�                                    Byo)2  �          @��H�h��@>�R�@���0{B�녿h��@0  �N�R�@ffB؞�                                    Byo7�  �          @��ÿTz�@7��C�
�6��B�(��Tz�@(Q��QG��GG�B��H                                    ByoF~  �          @�Q�fff@4z��C�
�7��B�B��fff@%�P���G�RB�8R                                    ByoU$  �          @������@9���C33�1�Bހ ����@*�H�P  �AG�B���                                    Byoc�  �          @��\��(�@<���;��*  B��쿜(�@.�R�HQ��9\)B�\                                    Byorp  �          @��\��p�@AG��5��#��B�#׿�p�@3�
�C33�3G�B��                                    Byo�  �          @�G���G�@9���8���)p�B��ÿ�G�@,(��E�8z�B�G�                                    Byo��  �          @�G����@0���>�R�0�\B��)���@"�\�J�H�?33B�3                                    Byo�b  �          @�����(�@'��K��>�
B�
=��(�@���W
=�Mz�B�=q                                    Byo�  �          @�=q�(�@.{�S33�G  B�Q�(�@�R�^�R�VBϮ                                    Byo��  �          @��;�33@(Q��`  �SQ�B�#׾�33@Q��k��cG�BĸR                                    Byo�T  �          @��\��
=@   �`���XB���
=@  �k��h�B�                                      Byo��  
�          @�zᾳ33@���i���a��BĞ���33@���s�
�q�Bƙ�                                    Byo�  �          @�{�
=q@$z��e�W\)B��
=q@�
�p  �f��B�G�                                    Byo�F  �          @����
@1��[��KG�B�\���
@"�\�g��Z�HB�G�                                    Byp�  �          @�\)��33@'
=�g��W��B�\��33@
=�r�\�g{Bģ�                                    Byp�  �          @��׾�G�@*�H�g��T�Bƙ���G�@�H�r�\�d33B�z�                                    Byp"8  �          @��H��G�@1G��g��Q  B�#׾�G�@!G��s33�`33B��f                                    Byp0�  �          @�(����@,���p  �W�HB�(����@(��z�H�g
=B��                                    Byp?�  �          @��\@.{�s33�X��B�
=�\@p��~{�g��Bų3                                    BypN*  �          @����33@6ff�~�R�X�HB�
=��33@%�����g�HB�z�                                    Byp\�  �          @��W
=@AG��xQ��PG�B���W
=@0����=q�_G�B��H                                    Bypkv  �          @�\)���
@G
=�xQ��M33B��׼��
@6ff���\�\(�B��3                                    Bypz  �          @�>�@J=q�q��Hz�B�k�>�@:=q�~�R�WQ�B�
=                                    Byp��  T          @�ff�8Q�@\)�h���Z�B�\)�8Q�@  �s33�h=qBי�                                    Byp�h  �          @��R���?�
=�xQ��s�
C �)���?�
=�~�R�~��C33                                    Byp�  �          @�����H?�\)�z�H�{�C� ���H?�\)��  �\CxR                                    Byp��  �          @�ff��=q?����Q�33C)��=q?O\)��=q�HC�R                                    Byp�Z  �          @�Q���?p�����Hp�C���?.{��z���CW
                                    Byp�   �          @�p���33?����Q�z�C�\��33?�����\�C��                                    Byp�  �          @�\)�J=q?���vff�HB�{�J=q?�33�{��B��H                                    Byp�L  �          @�(��aG�?u�z�H.C�=�aG�?8Q��~�R
=C�)                                    Byp��  �          @���Tz�?�33�vff� B��
�Tz�?h���z�H��Ch�                                    Byq�  �          @�{�.{?�  �tz�=qB�R�.{?��\�z=q� B�u�                                    Byq>  �          @��׾\?�Q��q��y��B�G��\?��H�x����B�8R                                    Byq)�  �          @�p���G�?����n�R�}p�B�(���G�?˅�u�ffB��H                                    Byq8�  T          @��;�?�z��q��
B�k���?�
=�w�aHB�B�                                    ByqG0  �          @�{��G�?�\)�uBҞ���G�?���{�L�B�k�                                    ByqU�  �          @��þ��?�\)�|(�\BЙ����?���������B�=q                                    Byqd|            @�\)�333?����{��B��333?����Q���B�                                    Byqs"  T          @�녿��\?�=q�vff���B��\���\?�{�|(�ffB��                                    Byq��  
�          @��H��\@�������B�\)��\?�z���L�B��                                    Byq�n  T          @�녾�p�@+������f�B��H��p�@���p��s��Bŀ                                     Byq�  �          @��R��{@0�����\�^��B�\��{@!G���\)�l  B�aH                                    Byq��  T          @�ff�5@7
=�j�H�L��BϸR�5@)���u��Y��B��)                                    Byq�`  �          @���@0�������]��B��f��@!G���ff�jz�BȞ�                                    Byq�  �          @����@3�
���H�\z�BƏ\��@%�����i=qB�8R                                    Byq٬  "          @�  �\)@0����(��^�
B�  �\)@!G������kffB�{                                   Byq�R  "          @��R���@-p���(��a�B�Q���@{�����n�\B��                                   Byq��  �          @�
=��\@(����p��d�B�녿�\@=q����q�B�                                    Byr�  �          @�ff�\)@'�����e
=B�{�\)@Q������qffB�\)                                    ByrD  "          @��
��@#33����fp�B˞���@z�����r�RB��H                                    Byr"�  "          @�=q�Ǯ@$z������eG�B�aH�Ǯ@��{�q��B�                                    Byr1�  
�          @��H��p�@(Q���G��b��B�  ��p�@=q���o{B�u�                                    Byr@6  "          @����{@)������b�HB{��{@�H��ff�o�B��                                    ByrN�  
=          @�z�\@,(���=q�aQ�B�(��\@p����R�mz�Bř�                                    Byr]�  ?          @�\)�\@2�\����^{B�ff�\@$z���  �j33Bĳ3                                    Byrl(  	�          @�ff��Q�@5��G��Z��B�\)��Q�@'����f�HBÊ=                                    Byrz�  T          @�p���p�@333�����[�B����p�@%���p��g�B�\)                                    Byr�t  �          @��R����@:=q��  �V��BÊ=����@,����z��b�BĽq                                    Byr�  
�          @���Q�@:�H�|���U(�B�
=��Q�@-p����H�a  B��                                    Byr��  
�          @��
�\@7
=�z�H�V��B�Q�\@)����=q�bffBĀ                                     Byr�f  
�          @����@333�|(��Y
=BƳ3��@%���\�d�RB�.                                    Byr�  �          @�=q��@+��~{�^
=B�{��@{����i��B��H                                    ByrҲ  
�          @�
=��@+���(��a��B��H��@{�����m=qB��H                                    Byr�X  �          @����@H���w��K��B�G���@<(������Wp�B��{                                    Byr��  	�          @�\)��@@���~{�SffB�p���@333���
�_  B�                                    Byr��  
�          @��ý�Q�@/\)�z=q�[��B��q��Q�@"�\��G��gz�B�
=                                    BysJ  
�          @�\)�u@=p���  �U��B�G��u@0  ��z��ap�B�p�                                    Bys�  "          @��ͽ#�
@333��Q��\33B��#�
@&ff��z��g��B��f                                    Bys*�  "          @�=q<��
@(���n�R�Z�
B�W
<��
@p��w
=�f33B�G�                                    Bys9<  
Z          @��׾u@7��s33�SQ�B����u@+��|(��^��B�G�                                    BysG�  T          @�ff��(�@<(��z=q�S\)BĀ ��(�@/\)�����^z�BŨ�                                    BysV�  7          @�
=�u@.{�s33�Y��B�ff�u@!��{��dB��{                                    Byse.  
E          @�
=?L��@���k��^z�B��?L��@p��r�\�i  B�Q�                                    Byss�  
�          @�\)?�(�?�(��u��kG�Biz�?�(�?���z�H�t��B_
=                                    Bys�z  �          @���?�z�?�z��mp��effBU��?�z�?�p��r�\�n{BJ�                                    Bys�   
�          @��
?aG�@Q��l���h{B�?aG�?��H�s33�r(�B�\)                                    Bys��  �          @�z�?��H@�
�hQ��bz�Bn33?��H?���n�R�kBe�                                    Bys�l  "          @���?�
=@���c�
�X��B`��?�
=?�(��j=q�a�BW��                                    Bys�  
�          @�z�?�p�@��c33�X
=B\ff?�p�?����i���`BSG�                                    Bys˸  �          @���?���@G��j�H�b  Ba{?���?��p���j��BW\)                                    Bys�^  
�          @�ff?���@��n{�c33Ba?���?����s�
�l  BX                                      Bys�  
�          @��?���@33�j�H�[Q�BR=q?���?���p���c�BHff                                    Bys��  �          @�?�{@��g
=�Z{Bg?�{@ ���l���b��B_G�                                    BytP  "          @��H?�ff@{�dz��]��B�L�?�ff@�
�j�H�g�B{�                                    Byt�  "          @��R?�@   �Z=q�Gp�Bq�?�@�aG��P�Bj�                                    Byt#�  
�          @�?�33@�\�n{�g  Br{?�33?�\)�s�
�p
=Bi=q                                    Byt2B  "          @��?�\@\)�fff�[
=B��H?�\@z��mp��e=qB�G�                                    Byt@�  
Z          @�(�=L��@\)�dz��\p�B��{=L��@��k��f�
B�k�                                    BytO�  �          @���?�(�@=q�b�\�M��Bj=q?�(�@\)�h���V�Bc                                      Byt^4  �          @�
=?�ff@G��g
=�X=qBpff?�ff@ff�l���a�Bh��                                    Bytl�  �          @��?\)@G��i���eG�B��?\)@
=�p  �o=qB�u�                                    Byt{�  
Z          @��?0��@ff�e�^��B�W
?0��@��l(��hB�
=                                    Byt�&  
�          @��;�z�@(���^�R�R�B�aH��z�@�R�e�\��B�#�                                    Byt��  �          @��>u@�R�mp��kQ�B��{>u@z��s�
�uffB��{                                    Byt�r  T          @��?
=q?���xQ��|\)B�Q�?
=q?�(��}p�
=B�Q�                                    Byt�  
�          @��?�@�\�o\)�rp�B�?�?�\)�u��|(�B�L�                                    Bytľ  �          @�33?z�@ ���tz��u\)B���?z�?��y���~��B�.                                    Byt�d  
�          @��
?G�@��r�\�pp�B�\?G�?�\)�w��y�RB���                                    Byt�
  �          @��H?W
=?����u�yz�B�� ?W
=?�33�z=q8RB|G�                                    Byt�  �          @�(�?L��@Q��n{�j33B��
?L��?��H�s�
�sp�B��q                                    Byt�V  "          @�?fff@��dz��X33B��?fff@G��j�H�ap�B��\                                    Byu�  �          @�33?fff@�R�hQ��a�B�33?fff@z��n{�k
=B�33                                    Byu�  �          @�33?L��@���l���i�B��)?L��?�p��r�\�rG�B��
                                    Byu+H  T          @��\?h��@�\�mp��l  B���?h��?���s33�t�HB�                                    Byu9�  �          @���?��@ ���i���h\)Bx�?��?����o\)�p�Bp                                    ByuH�  T          @��?��?�Q��l���l�Bt�?��?��
�q��u�Blp�                                    ByuW:  
�          @��H?p��?�\)�s33�t�\B}�?p��?��H�w��}(�Bt��                                    Byue�  
�          @���?@  ?���s33�w��B�� ?@  ?�(��w�\)B��
                                    Byut�  
Z          @��?!G�?�Q��r�\�vB�#�?!G�?��
�w���B��                                    Byu�,  T          @�G�?.{?�(��p  �s�
B�  ?.{?��u��|�B��                                    Byu��  T          @�=q?
=?�(��s33�vp�B�� ?
=?�ff�xQ����B��3                                    Byu�x  �          @��H?�@	���n{�l��B�33?�?��R�s�
�u�
B���                                    Byu�  T          @��?.{@p��l(��g�HB��\?.{@33�q��p��B��                                    Byu��  �          @�33?+�@�
�p���pffB��?+�?�33�vff�yp�B��H                                    Byu�j  �          @�33?��?�
=�w
=�z�B��f?��?�G��{���B�#�                                    Byu�  T          @��?��?�33�xQ��|{B���?��?�p��|���{B�Ǯ                                    Byu�  T          @�p�?&ff?޸R��Q�(�B�  ?&ff?�=q���\�\B�\                                    Byu�\  T          @�>�ff?�{�~�R�{B�#�>�ff?ٙ����(�B���                                    Byv  
�          @���>.{?�����(�Q�B�{>.{?�33��ffB���                                    Byv�  T          @��=#�
?�������=qB�k�=#�
?�33���H�B�.                                    Byv$N  
(          @�p�=u?�ff��G���B�u�=u?У������B��                                    Byv2�  �          @�{>��?�33���H��B��f>��?�p����W
B��
                                    ByvA�  T          @�p�>��R?�����H� B��=>��R?�(�����{B�(�                                    ByvP@  �          @�z�?��@Q��u��]p�B��?��@{�z�H�e��B{�                                    Byv^�  �          @��H?��@(���[��A=qBn�H?��@   �b�\�I=qBiQ�                                    Byvm�  �          @�Q�?p��@z��mp��`
=B�?p��@
=q�s33�h��B�=q                                    Byv|2  �          @�
=?=p�@
�H�u�lB�L�?=p�@ ���{��uz�B��{                                    Byv��  �          @�  ?8Q�@��y���p�B�.?8Q�?��H�~�R�yp�B�W
                                    Byv�~  �          @�?aG�@z��s�
�m�B��?aG�?��x���v  B���                                    Byv�$  �          @�?@  @���s33�l�RB�G�?@  ?��R�x���up�B��                                     Byv��  �          @�?   @  �qG��j=qB��H?   @ff�w
=�s=qB��                                    Byv�p  �          @���>��?�Q���ffB�B�{>��?�\�����)B�8R                                    Byv�  �          @��H�L��?���\)ǮB��׾L��?У�����\)B�
=                                    Byv�  �          @��
���?���Q�{B�=q���?У����\�B�B�                                    Byv�b  �          @�(���=q?���Q��BĔ{��=q?�z����\��B�\)                                    Byw   �          @��
��{?������#�B�8R��{?�{���H��Bˏ\                                    Byw�  T          @��B�\?�Q������B�=q�B�\?�\��=q33B�Q�                                    BywT  �          @��\��\)?�(��}p�� B�(���\)?Ǯ����
=B�                                      Byw+�  �          @�p���  ?�Q���=qffB��;�  ?��
��z��BƔ{                                    Byw:�  �          @�
=�8Q�?�33��=qk�B�Ǯ�8Q�?�(���z���B��)                                    BywIF  �          @����  @ff����~=qB�ff��  ?���=q�B�                                    BywW�  �          @����z�@33��Q�\B�  ��z�?�����H��B�p�                                    Bywf�  �          @�ff��  @   ���{B�p���  ?�=q�����B�                                    Bywu8  �          @��\���?�{�����B��þ��?�Q���\)G�B�W
                                    Byw��  �          @����?�(���(�#�B؊=��?����{u�B�ff                                    Byw��  �          @�����?޸R��{33Bؙ���?Ǯ��  �B�z�                                    Byw�*  �          @�  �   ?�(���G�Q�B�ff�   ?�ff���
�qB�                                    Byw��  �          @��׿�\@������p�B�p���\?�{��33(�B���                                    Byw�v  T          @��H�
=q?����33Bә��
=q?�  ��\)��B֣�                                    Byw�  �          @�����G�?�33��(���B����G�?�p����Rk�Bг3                                    Byw��  �          @��׾���?�33����fBˮ����?�p���aHB�                                      Byw�h  �          @���^�R?��H������B�z�^�R?��
���R��B�33                                    Byw�  �          @��׿�  ?z�H�����C�Ϳ�  ?L����ff#�C�\                                    Byx�  �          @�\)��p�?������Q�C\��p�?��\��33\C\)                                    ByxZ  �          @����Q�?����(�=qC�쿘Q�?�{��p�� C	
=                                    Byx%   �          @�G����H?�������z�CaH���H?��
��{G�CǮ                                    Byx3�  �          @��\����?�Q���{G�C�Ῑ��?�G���  �RCǮ                                    ByxBL  �          @�33����?��R��\)
=B��쿌��?��������C                                    ByxP�  �          @�33����?�ff���R��B�\)����?�\)�����C �{                                    Byx_�  
�          @��
��?�ff���R��B����?�\)�����\C�=                                    Byxn>  �          @�33����?�=q��ffB�B�aH����?�33����  C �                                    Byx|�  T          @��H���
?�{��{u�B�𤿃�
?�
=��  L�B�B�                                    Byx��  �          @�=q�p��?��������}�B�k��p��?��
��(��3B�Ǯ                                    Byx�0  �          @��\�n{?�
=���\�~�\B��n{?�  ����k�B��                                    Byx��  �          @�  ���\?333��  aHC{���\?�����ffC��                                    Byx�|  �          @��R��p�>��H��Q�=qCh���p�>���������C&!H                                    Byx�"  �          @����  >�
=��\)�fC!h���  >k����{C)�3                                    Byx��  �          @�G����H?
=q��Q��)C�쿺�H>�33����8RC&��                                    Byx�n  �          @�����33?:�H��
=(�C���33?
=q����
C!ٚ                                    Byx�  �          @��ÿ�?�G�����\C�׿�?Q����H(�C��                                    Byy �  �          @�z���
?����(��g�RCY����
?��H���R�nffC{                                    Byy`  T          @�z�ٙ�?�33��p��jffC��ٙ�?�(�����qG�C��                                    Byy  �          @��Ϳ��
?������t{C=q���
?У�����{�C.                                    Byy,�  �          @����=q?�\)��ff�n�C=q��=q?ٙ������u��C�                                    Byy;R  �          @�����H?�p���{�n�B��׿��H?������v
=C �H                                    ByyI�  �          @�(��\?�����  �r��Cc׿\?���=q�yCB�                                    ByyX�  �          @�z�Y��@#�
��=q�b{Bأ׿Y��@�����k{B�{                                    ByygD  	�          @���Y��@��p��l�RB�33�Y��@
=q��Q��u��B�(�                                    Byyu�  "          @���333@Q���p��l�
B�녿333@p���Q��v  B�ff                                    Byy��  T          @���(��@���\�|B�(��(��?�33����B�L�                                    Byy�6  T          @���\@Q���G��q�HB�{��\@����z��{=qB�                                      Byy��  �          @��;�Q�@"�\���i�HB�W
��Q�@
=�����sffBŏ\                                    Byy��  
�          @�zᾨ��@,(�����`��B�𤾨��@!G���p��j�\B��H                                    Byy�(  �          @�zῚ�H@����R�mz�B�����H@   �����u�HB�ff                                    Byy��  "          @�\)�^�R@���Q��k{B�Q�^�R@  ����t(�B�=q                                    Byy�t  �          @�{�#�
@���  �mBѣ׿#�
@\)��33�w(�B��                                    Byy�  �          @��Ϳ#�
@\)��p��iG�B��ÿ#�
@�
�����r�RB��                                    Byy��  "          @��(�@,(����H�_��Bͮ�(�@ ����ff�i�B�k�                                    Byzf  
Z          @�  ���@<(���Q��U��B��þ��@1G���z��_\)B�{                                    Byz  T          @��\��R@<�����H�VQ�Bˮ��R@1G����R�_��B�(�                                    Byz%�  �          @�z�^�R@{���m��B�녿^�R@G������v��B���                                    Byz4X  �          @�z�=p�@0�������`�
B��=p�@$z���(��jffB�                                    ByzB�  �          @��
�:�H@0  ��  �`��BѮ�:�H@#�
���
�j�\Bӽq                                    ByzQ�  �          @��Ϳ\(�@+�����c=qB׏\�\(�@\)��p��lB��                                    Byz`J  �          @���.{@5������X��B��ÿ.{@)����p��b�BиR                                    Byzn�  �          @��ÿ
=@?\)��Q��S  B�=q�
=@3�
��(��\�HB˞�                                    Byz}�  �          @�z�p��@+��\)�Z=qB��)�p��@   ��33�c�RB�k�                                    Byz�<  "          @���z�H@4z���  �TB�aH�z�H@(�����
�^Q�B�Ǯ                                    Byz��  
o          @�
=�aG�@>{�xQ��M��B���aG�@2�\��Q��WffB�
=                                    Byz��  ?          @�ff�@  @?\)�w��M�HB�\�@  @3�
��  �W��Bѽq                                    Byz�.  �          @��R�c�
@E�q��Fz�B��ÿc�
@:�H�z�H�P\)Bսq                                    Byz��  �          @�G����\@B�\�x���J{B����\@7
=�����S��B�#�                                    Byz�z  7          @��ÿc�
@:�H��  �R�\B���c�
@.�R���
�\p�B�=q                                    Byz�   
�          @�\)�+�@3�
��  �XB���+�@'���(��b�
B���                                    Byz��  
�          @�z�@  @2�\�|(��V�B�(��@  @'
=����`��B�(�                                    By{l  "          @�G��Tz�@AG��|���N�\B��ͿTz�@5����H�X��BԮ                                    By{  
�          @�  �W
=@333�����X��BՀ �W
=@'
=���b�
B�Ǯ                                    By{�  
�          @��!G�@(������b=qB��ÿ!G�@����\)�lz�B�                                      By{-^  "          @�Q�J=q@&ff��ff�dffB�Ǯ�J=q@����=q�n�B�aH                                    By{<  T          @�����@0  �r�\�L�
B�33����@$z��z�H�VffB�G�                                    By{J�  T          @�녿޸R@<���X���2B��޸R@2�\�aG��;�HB��)                                    By{YP  �          @�(��޸R@9���aG��8\)B���޸R@.�R�i���Az�B�
=                                    By{g�  
�          @��R��\@1��l���A=qB�녿�\@&ff�u��J=qB�aH                                    By{v�  T          @��H�޸R@%�l(��GQ�B����޸R@=q�s�
�P=qB�u�                                    By{�B  
�          @�(��У�@G��k��SG�B�z�У�@�q��\  B���                                    By{��  �          @�녿�{@&ff�fff�A�\B����{@�H�n{�JQ�B��                                    By{��  �          @��H����@/\)�w��F  B�
=����@#33��  �O  B��)                                    By{�4  
�          @�\)���H@/\)�r�\�FG�B����H@#33�z=q�O�\B��                                    By{��  �          @�  ��p�@%��z=q�N
=B��Ϳ�p�@�������W�B���                                    