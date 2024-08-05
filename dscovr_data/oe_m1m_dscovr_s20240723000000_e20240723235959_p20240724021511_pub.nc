CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240723000000_e20240723235959_p20240724021511_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-24T02:15:11.212Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-23T00:00:00.000Z   time_coverage_end         2024-07-23T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy�4@  
�          @�ff�c�
?L��@�G�B�k�C

�c�
?�ff@��B��Ch�                                    By�B�  "          @�p���G�<��
@�(�B�p�C)�R��G�>���@��
B���B��                                    By�Q�  
�          @��=��
=�G�@��HB��BS=��
>���@��\B�33B�8R                                    By�`2  
�          @�����Q�=L��@���B�33C+��Q�>�{@�  B�B�C                                    By�n�  "          @�����ff>�@�Q�B���CQ��ff?@  @�
=B��HB��
                                    By�}~  �          @��\��?�@���B��3C
��?\(�@�\)B��RB��H                                    ByΌ$  "          @�녿(�?Tz�@�p�B�z�B�ff�(�?�{@��
B�\B���                                    ByΚ�  �          @�=q�c�
?�33@��RB��{B�q�c�
?�z�@��B���B���                                    ByΩp  T          @���c�
?�(�@��
B��B�{�c�
?�  @���B�p�B���                                    Byθ  �          @�
=�W
=?�@�33B��HB�{�W
=?�Q�@�  B���B��f                                    By�Ƽ  �          @��
�k�@	��@���B{Q�B� �k�@��@�p�Bo(�B�                                      By��b  T          @�(��xQ�?�{@���B���B�\)�xQ�@
=@���Bv(�B��                                    By��  
C          @��׿��?��
@�{ByG�B�����@G�@��\Bn\)B�aH                                    By��  k          @�p����
@�\@��Bxz�B�k����
@�@�{Blz�B�\)                                    By�T  "          @�(����?z�@���B�G�CQ쿱�?\(�@�Q�B�8RC@                                     By��  �          @�����H�c�
@��\B�u�CX&f���H���@��
B���CNff                                    By��  "          @�Q�<����H@�ffB�ǮC���<���@�G�B�� C��                                    By�-F  "          @�G��aG��s33@�z�B�Q�C���aG��+�@�{B��)C|�                                    By�;�  T          @��H��33�u@��RB��Cz���33�.{@�Q�B�Cr��                                    By�J�  "          @��H�8Q�L��@��RB�#�Cd
�8Q��@�  B��qCW��                                    By�Y8  
�          @��\�u��G�@��RB�B�C�Z�u�8Q�@�Q�B��)C{�=                                    By�g�  "          @�33��G��!G�@���B�ǮC��q��G���{@��\B�z�C{�                                    By�v�  
�          @��׾#�
��p�@��B�33Cw33�#�
��Q�@�  B���CQ)                                    Byυ*  	�          @�
=�\��@�p�B�W
Ci�H�\�k�@�{B�(�CS�                                    Byϓ�  �          @�(����þ��@�=qB�  Cj����þ#�
@��HB��CN�                                    ByϢv  
�          @�G��(��Tz�@�B�u�Ci�R�(���@�
=B���C\Ǯ                                    Byϱ  
�          @������R@���B��CeG�������@��B�#�CS\)                                    ByϿ�  
Z          @�p��&ff��  @��HB�z�CHٚ�&ff=�\)@�33B�k�C-�                                    By��h  "          @��\>���0��@�  B��\C��>����p�@���B�k�C��                                    By��  T          @�=q�#�
�.{@��RB��RCb���#�
��33@��B�(�CP��                                    By��  �          @���>k��aG�@�{B��C�C�>k����@�\)B���C�7
                                    By��Z  
�          @����Ϳ   @���B�
=C^���;.{@�=qB��qCE)                                    By�	   
�          @��k��#�
@��HB��C=�ÿk�>8Q�@��HB�
=C(޸                                    By��  =          @��k�>B�\@��HB�
=C(0��k�?�@�=qB�ffC�                                    By�&L            @�G���=q��z�@��
B�� C@(���=q=u@��
B��C1��                                    By�4�  �          @�=q��{>�z�@�=qB��qC)ٚ��{?!G�@�G�B��RC�f                                    By�C�  
�          @���\)��\)@���B�#�CNJ=�\)=��
@��B�W
C+��                                    By�R>  
�          @��ÿ�ff>��R@�z�B��C#����ff?(��@��B���C�=                                    By�`�  T          @���Q�>Ǯ@�G�B��=C�q�Q�?8Q�@�  B�ffC
�                                    By�o�  
�          @����#�
�(��@�
=B�{C�*=�#�
��p�@�  B���C��3                                    By�~0  �          @�논#�
�Q�@�
=B�#�C��H�#�
��@���B��3C�:�                                    ByЌ�  T          @���>��H����@��B��HC�>��H�h��@�B�ǮC�q                                    ByЛ|  
�          @��\?#�
�}p�@�p�B�33C��f?#�
�0��@�
=B�=qC�|)                                    ByЪ"  �          @��H=u�+�@���B�.C���=u��Q�@��B���C���                                    Byи�  "          @�����{�   @�  B���Ck�׾�{�L��@���B�\CR.                                    By��n  
�          @�  ��\�L��@��RB���C9����\>�  @�ffB�aHC�\                                    By��  
�          @�ff���u@���B��HCs+���=u@���B�z�C}q                                    By��  T          @���=��;�{@�Q�B�
=C�xR=��ͼ�@���B�L�C�L�                                    By��`  
�          @��    �k�@���B�(�C��    =��
@��B��B��                                    By�  
Z          @��<���  @���B��C��
<�=�\)@��B�(�B�
=                                    By��            @�z�>8Q쾊=q@��B�C��H>8Q�=L��@��
B�\)A�
=                                    By�R  �          @��\�B�\<��
@�=qB��C/�\�B�\>���@���B�L�B�(�                                    By�-�  �          @�=q�#�
���R@��B�8RC���#�
<#�
@�=qB��=C�R                                    By�<�  T          @��H>�����
=@���B�  C�q�>�������@�=qB�#�C�y�                                    By�KD  T          @��>��z�@��HB�\)C��3>���  @��
B���C��q                                    By�Y�  T          @�ff?h�ÿE�@�G�B��\C��?h�þ�ff@��\B��C��                                     By�h�  
�          @���?5���@�B�{C��?5��Q�@�ffB��qC�t{                                    By�w6  9          @���?s33�L��@�{B�ffC�  ?s33=�@�ffB��3@�ff                                    Byх�  
�          @���?:�H��=q@�\)B�=qC�  ?:�H=u@�\)B�aH@�33                                    Byє�  
�          @���?+�����@�ffB�L�C��f?+�<�@��RB��@	��                                    Byѣ(  "          @��>Ǯ�L��@��RB��C�>Ǯ>�=q@�ffB���B\)                                    Byѱ�  
�          @�=q>��#�
@�G�B�
=C��
>�>���@���B�u�B
ff                                    By��t  T          @��?
=q>L��@��B��3A��
?
=q?
=q@���B���B4ff                                    By��  
�          @�33>��\)@���B��C���>�>L��@���B���A�\)                                    By���  T          @��\>\=�Q�@��B�{AV=q>\>�(�@�G�B�u�BA�                                    By��f  
�          @��>8Q�>�@�G�B�z�Bff>8Q�>��@���B�Q�B���                                    By��  �          @�p��#�
=�@��B�
=B����#�
>��@�z�B�W
B�p�                                    By�	�  
�          @�G�>�\)��Q�@���B�
=C�R>�\)>�\)@���B���B3G�                                    By�X  
�          @�Q�Ǯ>�Q�@�ffB��C�H�Ǯ?8Q�@��B�\B�u�                                    By�&�  
�          @��H���R>�{@�G�B�z�C����R?0��@�  B���B��                                    By�5�  �          @��H��\)?�@���B���B�׾�\)?fff@��B�ffB֣�                                    By�DJ  
�          @��þ.{>���@�Q�B��qB��.{?+�@�\)B�k�B�ff                                    By�R�  T          @���B�\>�p�@��HB��3B�3�B�\?E�@�G�B�=qB�k�                                    By�a�  "          @��
=#�
>�@��HB��RB�{=#�
?\(�@�G�B��RB�=q                                    By�p<  "          @���>��>��
@��B��B	p�>��?=p�@�ffB��HBe(�                                    By�~�  "          @�{>�p�?E�@�33B���B��
>�p�?�@���B�z�B��                                    Byҍ�  "          @������
?�Q�@���B�B�B�\)���
?�=q@���B��B�                                      ByҜ.  
(          @��H��{?�(�@���Bf��C\)��{@33@��BY{C �)                                    ByҪ�  "          @���
?�=q@y��Baz�C�=��
?��@p��BU�\C	}q                                    Byҹz  �          @�(��B�\?���@���B��B��=�B�\?��R@�G�B�z�B�
=                                    By��   �          @�ff���R?z�@�z�B���C����R?p��@��\B��qC��                                    By���  "          @�Q���
>\@�z�B�.C'�{���
?B�\@��HB�Q�C�                                    By��l  "          @�z���Ǯ@�\)B�8RC@��    @��B�8RC3޸                                    By��  �          @�ff��(��\)@��
B�B�CF  ��(����@���B�(�C9{                                    By��  �          @����.�R>�z�@�z�Ba�RC-��.�R?!G�@�33B^�C&��                                    By�^  �          @��
�1G�?c�
@xQ�BT��C"
�1G�?�(�@r�\BM��CB�                                    By�   	�          @����?!G�@�{Bp=qC$�
��?z�H@�(�Bi�C!H                                    By�.�  �          @��H���?#�
@�\)BsC$&f���?�  @��Bm�C.                                    By�=P  �          @����Q�>���@���B��RC)O\�Q�?J=q@�
=B|33C                                    By�K�  �          @�Q��p�>��@��B(�C0&f�p�?�@��RB|(�C&n                                    By�Z�  "          @�  ����@�G�B���C7J=��>�=q@�G�B�z�C,                                    By�iB  �          @�zῃ�
?�\@�Q�B�k�C�=���
?u@�ffB��C�                                    By�w�  	�          @��H��p���@��B�W
C9�῝p�>�{@���B�z�C$\)                                    Byӆ�  �          @�  ��33��\@��B��RCK�H��33���
@�(�B��C4�\                                    Byӕ4  "          @�Q쿢�\=�@��B�\)C.�{���\?(�@��\B���Cn                                    Byӣ�  �          @�����Q�#�
@�p�B�8RC5�Ϳ�Q�>�@���B�u�C�                                     ByӲ�  T          @�
=������@��B���C:\)���>�33@���B��C%��                                    By��&  �          @��G��aG�@���B�(�C:)�G�>��@���B�\C,��                                    By���  
�          @�\)��\)�\)@�z�B���C8aH��\)>�{@�(�B�{C)��                                    By��r  
Z          @���z�=���@�33B��RC1!H��z�?z�@�=qB��C#�                                    By��  �          @����  >u@��
B��C,=q��  ?:�H@��\B��Cs3                                    By���  �          @��H��>�G�@�\)B��RC'0���?h��@�p�B�B�C�
                                    By�
d  
�          @�������=�\)@�\)B�C1�)����?��@�ffB�L�C#xR                                    By�
  �          @������H>�ff@�  B��qC%Y����H?n{@�{B���Cz�                                    By�'�  
�          @�\)��p�>8Q�@�B��C.+���p�?(��@�z�B�p�C33                                    By�6V  T          @�=q��H�u@�{Bz33C9�{��H>L��@�{Bz\)C/@                                     By�D�  �          @�z��ff>���@���B�B�C)�H��ff?L��@�  B�\C��                                    By�S�  "          @�Q�޸R>�{@�
=B�p�C(�{�޸R?Tz�@��B���Cff                                    By�bH  T          @�z���>�(�@���B�k�C$p����?k�@��\B��CT{                                    By�p�  T          @�녿�?fff@�  B��qC�=��?�\)@�z�B���C!H                                    By��  �          @�G���Q�?�@���B�� C�3��Q�?��@�z�Bu�C\                                    ByԎ:  T          @��H�(�?�\)@g�BL
=C���(�?��H@\(�B?  CL�                                    ByԜ�  T          @��׿�(�?Tz�@�{B��C���(�?�G�@��HB��C5�                                    Byԫ�  �          @���$z�>��@~{Bc�RC.E�$z�?#�
@z�HB_�HC%�R                                    ByԺ,  T          @����P  =�@`  B<Q�C1�f�P  >�@^{B:Q�C+�                                    By���  �          @�
=�;�>u@j�HBM{C/Q��;�?��@hQ�BI�HC(s3                                    By��x  j          @�  �?\)>���@j�HBJp�C-���?\)?0��@g�BF��C&��                                    By��  	�          @�=q�G�>���@~�RBO=qC.���G�?0��@{�BK�C'}q                                    By���  T          @��H�Dz�>���@���BR=qC-���Dz�?:�H@~�RBNQ�C&�=                                    By�j  T          @���;�>.{@�B[�C0�
�;�?
=@�z�BX�\C(�\                                    By�  �          @��
�U>u@w�BD\)C/޸�U?!G�@tz�BA\)C)Q�                                    By� �  "          @�=q�Mp�>�ff@x��BH�RC+��Mp�?Y��@tz�BD{C%8R                                    By�/\  
�          @����E?E�@x��BJ�C&
�E?�z�@s33BC��Cs3                                    By�>  �          @�ff�7
=?.{@j=qBL��C&�{�7
=?�ff@dz�BF�C�                                    By�L�  �          @�=q�*�H?0��@hQ�BR��C%���*�H?��@b�\BK��Cn                                    By�[N  �          @�{�Fff?(��@^{B>G�C(
=�Fff?�G�@XQ�B8�C!�                                    By�i�  T          @��H�L(�    @Q�B7\)C3���L(�>�{@P��B6
=C-�
                                    By�x�  �          @�
=�XQ�#�
@7�BG�C>��XQ쾮{@:�HB"�RC9�q                                    ByՇ@  "          @�  �Q�?E�@Y��BV�C!�q�Q�?���@S33BM��C��                                    ByՕ�  "          @���� ��?�
=@g�B^�C��� ��?�@\��BO�C

=                                    Byդ�  �          @�p���p�?���@j�HBh�\CaH��p�?�G�@a�B[
=C�3                                    Byճ2  �          @��H��R=L��@]p�BYC2����R>�
=@\(�BWz�C*k�                                    By���  �          @�녿��H>�p�@s33By
=C)ff���H?E�@o\)Br33C��                                    By��~  �          @��H��(��#�
@�(�B��
C55ÿ�(�>���@��Bp�C(s3                                    By��$  "          @��\��(�>B�\@�\)B��=C-�Ϳ�(�?&ff@�B��{C8R                                    By���  
�          @�����(���p�@���B��CB.��(�=�\)@�G�B��C1O\                                    By��p  T          @��ÿ�G���G�@�33B��3C8����G�>�{@��\B��\C$�{                                    By�  T          @��H��\)?333@xQ�Bz�C����\)?�\)@q�Bo�C\                                    By��  T          @�\)��  ?W
=@�33B���C�f��  ?��
@\)B~��Cp�                                    By�(b  �          @�녿��ü��
@~{B��C4�����>�
=@|��B�Q�C$�R                                    By�7  T          @��\��׾���@x��Bn�C;�����>\)@y��Boz�C0��                                    By�E�  �          @��R���>�ff@e�BjG�C(:����?Y��@`��Bb�
Cs3                                    By�TT  
�          @�����33>�\)@uB}��C+�ÿ�33?5@r�\Bwz�C�                                    By�b�  
>          @��� ��>B�\@u�Bx�C.�� ��?!G�@q�Bs�RC"�                                    By�q�  	�          @�p���ff>�p�@n�RB  C(����ff?J=q@j�HBw
=C^�                                    ByրF  
Z          @�Q��>�@k�Bo  C'0��?fff@fffBf�
Cٚ                                    By֎�  
Z          @�G���
?0��@j=qBm�C!h���
?�{@c33Bb�\C��                                    By֝�  T          @���1�?�R@N�RBB�C'� �1�?}p�@H��B;Q�C s3                                    By֬8  
�          @����-p�?xQ�@L(�B?�C 8R�-p�?��@C33B5
=C�R                                    Byֺ�  �          @�ff�+�?Ǯ@`��BB  C��+�?��H@S33B3G�C�\                                    By�Ʉ  
�          @�p��  ?��@dz�BJ(�C�f�  @33@Tz�B7CY�                                    By��*  
�          @�33����?���@��HBC�����?�@x��Bj�
B���                                    By���  �          @����
@6ff@B�\B4�\B��Ϳ��
@L(�@+�BB���                                    By��v  T          @����
=@G�@5B*  C���
=@%@#33B\)C0�                                    By�  �          @�33���?�z�@aG�BY��C�R���@�
@P��BC��B��                                    By��  
�          @��Ǯ@�@a�BSffB�(��Ǯ@%@P  B<G�B��                                    By�!h  "          @���
=@1G�@R�\BD��B�\�
=@I��@;�B)�\B�B�                                    By�0  T          @��&ff@:�H@:=qB0B�녿&ff@P  @!�Bp�B�p�                                    By�>�  T          @��Ϳp��@4z�@0��B+�B��p��@HQ�@��BB�G�                                    By�MZ  �          @�G��\)@G
=@\)B�
B�B��\)@X��@A�G�B���                                    By�\   �          @}p���p�@C�
@ffBQ�B�녾�p�@U�?��HA�Q�B���                                    By�j�  
�          @|(����
@(�@+�B.(�C  ���
@ ��@��Bp�B���                                    By�yL  
�          @p���O\)�
=?�G�A���C>Q��O\)��p�?���A�33C:s3                                    Byׇ�  �          @mp��+�>�ff@ ��B*��C*���+�?B�\@�B$�C$B�                                    Byז�  T          @u��\)?�\)@1G�BH{B�ff��\)@��@ ��B0�B�R                                    Byץ>  �          @�G��J=q@p�@QG�BY(�Bۀ �J=q@'
=@>{B=��B��H                                    By׳�  �          @|�Ϳ�
=?�{@Q�B`p�B��׿�
=@��@AG�BG�B�                                      By�  �          @��H�s33@G�@Z�HBcG�B�aH�s33@(�@HQ�BHQ�Bޏ\                                    By��0  �          @��
��z�?��@X��B]{B�����z�@33@G�BD�B���                                    By���  �          @�  ��  ?�@e�Bep�B�\��  @
=@S33BK�B�                                    By��|  
�          @�33��=q?�\)@_\)Bi=qB�  ��=q@33@N{BN�
B�(�                                    By��"  
p          @��
���?�{@_\)Bg�
B��f���@33@Mp�BM�B�3                                    By��  �          @�=q��  ?�@Z=qBc��B�k���  @G�@H��BJ  B��R                                    By�n  
(          @�녿n{?�@`  Bn
=B��Ϳn{@�@N�RBR��B�z�                                    By�)  
�          @�G��
=?�p�@w�B��BٸR�
=@{@g
=Bf
=Bѽq                                    By�7�  
�          @�z�0��?�33@x��BzQ�B���0��@��@g
=B]Q�B�                                      By�F`  T          @���?�@�  B��BԀ ��@ff@n{Bd=qBͣ�                                    By�U  "          @�  ��p�?�@uB��\B�\��p�@�
@c�
Bb�HB�33                                    By�c�  T          @xQ�5?��@S33BjQ�B݀ �5@�
@AG�BM�B�W
                                    By�rR  
�          @n{��@�@B	G�C �q��@(Q�?�  A�\)B�=q                                    By؀�  
�          @mp���{@*=q@(�B�RB���{@;�?�ffA�\)B��                                    By؏�            @o\)����@0��@��Bz�B�������@A�?�ffA��B���                                    By؞D  
�          @|�Ϳ��@\��?��A�(�B�  ���@g
=?h��AX(�BՏ\                                    Byج�  
�          @�33���
@J�H@�B=qB��ΰ�
@\(�?�p�Aə�B��
                                    Byػ�  T          @��H��  @`��?�
=A�  B�{��  @mp�?�
=A�  B�#�                                    By��6  "          @s33�:�H@A�@Q�B	p�B���:�H@S33?�Q�AԸRB��H                                    By���  
�          @n{��{@3�
?��RB33B��쿮{@C�
?�=qA�=qB�{                                    By��  
�          @i����@'
=@\)B,=qBʞ���@;�@ffB��B�33                                    By��(  
�          @dz�=�@1G�@{B��B��=�@C�
?�A�
=B��\                                    By��  
Z          @g
=��@5?�p�B	�HB�\��@E?ǮA��
B�aH                                    By�t  T          @qG�����@G�@�B
=B������@XQ�?�\)A��HB�B�                                    By�"  T          @r�\>#�
@P��?�33A�B�aH>#�
@`  ?�A�Q�B�                                    By�0�  
�          @o\)<��
@N�R?���A�(�B�Q�<��
@]p�?�\)A�Q�B�\)                                    By�?f  �          @p�׽#�
@O\)?�33A�B�ff�#�
@^{?�A�p�B�L�                                    By�N  "          @vff>��H@c33?�\)A�z�B�(�>��H@k�?
=Ap�B��                                    By�\�  T          @|(�>��R@l��?�ffA��RB�z�>��R@vff?@  A0z�B��
                                    By�kX  �          @tz�>k�@g�?�(�A���B��R>k�@p  ?.{A#�B�                                      By�y�  T          @n�R>���@Z=q?�Q�A�Q�B��f>���@e?n{Af�RB�p�                                    Byو�  "          @i��>�z�@\(�?�A�(�B�G�>�z�@dz�?&ffA%��B���                                    ByٗJ  "          @_\)>�(�@N�R?��HA�
=B��
>�(�@XQ�?8Q�A?�B�z�                                    By٥�  T          @k�?�@S�
?��RA�(�B��
?�@_\)?}p�Ayp�B���                                    Byٴ�  "          @g�?G�@P��?�{A��\B�8R?G�@[�?\(�A[�B�z�                                    By��<  
�          @_\)?#�
@N�R?��A�{B��{?#�
@W
=?
=A�B�ff                                    By���  �          @Z=q?z�H@AG�?�p�A�p�B��f?z�H@J�H?B�\AO33B��                                     By���  T          @\��?�=q@333?�G�AЏ\B��?�=q@?\)?�=qA��B�{                                    By��.  
�          @w�?���@E�?��HA�(�Bz\)?���@S33?�p�A��
B�ff                                    By���  
�          @n�R?�@>�R?�{A�p�B�
=?�@N�R?��A�  B�{                                    By�z  �          @`  ?z�@0��@�\Bz�B�33?z�@A�?˅A�  B�W
                                    By�   
�          @i��=�@9��@
�HB
=B�L�=�@L��?ٙ�A߅B��q                                    By�)�  �          @p  >\@7
=@�B�B�Ǯ>\@L(�?�33A��RB�W
                                    By�8l  �          @n{>���@%@!�B0G�B�W
>���@<��@ffB�B��                                    By�G  "          @k�����?�33@
�HBAffB��;���@{?���B{B�
=                                    By�U�  �          @j=q?#�
@p�?�(�B  B���?#�
@.�R?ǮA�=qB�z�                                    By�d^  p          @vff?�
=@8��?��RA�p�B~�?�
=@J�H?�G�A�(�B��                                     By�s  �          @~{�\)@ ��@Dz�BJ�B�L;\)@=p�@(Q�B&z�B�W
                                    Byځ�  
Z          @}p�?O\)@A�@��B=qB�B�?O\)@Vff?�G�A��B���                                    ByڐP  �          @�녾#�
@'�@EBF��B��
�#�
@E�@(��B"(�B���                                    Byڞ�  �          @�����G�@#�
@H��BKG�B�\��G�@A�@,(�B&��B�G�                                    Byڭ�  j          @������
@6ff@7�B4�B�#׽��
@QG�@�B�
B��q                                    ByڼB  
�          @���=�G�@=p�@.�RB*Q�B��)=�G�@W
=@p�B\)B�W
                                    By���  >          @��>�=q@A�@,(�B%�B���>�=q@Z�H@
=qB �B��                                    By�َ  
�          @���?!G�@L��@#33B�
B���?!G�@dz�?��RA�Q�B��)                                    By��4            @��
?�  @qG�@��B �B�G�?�  @��
?�A��RB��3                                    By���  
p          @��\?Y��@^{@8Q�B�B��?Y��@x��@�A�  B�k�                                    By��  
�          @���?O\)@K�@1�B!Q�B�Q�?O\)@e@{A�33B��                                    By�&  �          @��\?E�@H��@8Q�B&�RB���?E�@dz�@z�BB���                                    By�"�  
�          @��R?\(�@Z�H@.{B��B��?\(�@tz�@�A�p�B�Ǯ                                    By�1r  �          @�=q?���@`  @0  BB�\?���@y��@Q�A�  B�p�                                    By�@  �          @��H?k�@\��@9��B(�B�  ?k�@xQ�@�A��
B�8R                                    By�N�  T          @��\?u@Q�@B�\B&�B�aH?u@o\)@��Bz�B�8R                                    By�]d  �          @��
?�  @P��@FffB){B��
?�  @n�R@ ��B�
B���                                    By�l
  �          @���?\(�@@��@QG�B9  B�(�?\(�@aG�@-p�Bp�B��{                                    By�z�  �          @�{?0��@9��@<(�B2��B�?0��@Vff@=qB�\B��{                                    ByۉV  T          @�ff>��H@8��@h��BL��B�Ǯ>��H@]p�@EB%�B��                                    Byۗ�  T          @�Q�>���@=p�@�=qB]p�B���>���@i��@p  B6ffB��=                                    Byۦ�  T          @�\)?G�@=p�@�\)BX��B�W
?G�@h��@j�HB233B��R                                    By۵H  T          @�
=?(��@B�\@�{BUz�B�\)?(��@n{@fffB.��B��q                                    By���  �          @�
=?
=q@Dz�@�{BU33B��?
=q@p  @eB-�HB���                                    By�Ҕ  "          @�>�p�@>�R@��RBY�B���>�p�@j=q@hQ�B233B�p�                                    By��:  T          @�\)��@1G�@�Bg�B��f��@_\)@xQ�B?�
B��H                                    By���  T          @��R=�G�@333@�(�Bez�B�� =�G�@a�@tz�B==qB�k�                                    By���  "          @��?�\@2�\@��Bb��B�=q?�\@`  @p��B:�RB�ff                                    By�,  
Z          @��
>�@2�\@�Q�Ba�B�p�>�@`  @l��B9z�B�#�                                    By��  T          @��?!G�@1�@��Bb(�B��?!G�@`  @p  B:33B���                                    By�*x  T          @�?z�@7
=@���B^B�G�?z�@e�@l��B6z�B��R                                    By�9  
�          @��?�@:�H@�
=B[\)B���?�@hQ�@hQ�B2��B��=                                    By�G�  �          @�ff?�\@=p�@�\)BZ��B��=?�\@j�H@h��B1�HB�=q                                    By�Vj  
�          @�?Tz�@6ff@��B\
=B�aH?Tz�@c�
@j=qB4  B��                                    By�e  
�          @��?^�R@,��@��Bb��B�z�?^�R@\(�@p  B:��B��R                                    By�s�  T          @��?J=q@&ff@�z�Bh�
B�{?J=q@Vff@uB@�B�aH                                    By܂\  T          @�z�?���@"�\@��HBf\)B�L�?���@Q�@s33B?
=B��                                    Byܑ  "          @��?Ǯ@ ��@��RB[��Bhff?Ǯ@N�R@l(�B7{B��                                     Byܟ�  T          @��?��@Q�@���BYQ�BT�?��@Fff@i��B6
=Bo�                                    ByܮN  �          @�?���@�
@�33Bd�\B\��?���@Dz�@vffB@33By�R                                    Byܼ�  �          @��R?��H@'�@��HBa�RB�� ?��H@W�@r�\B:Q�B���                                    By�˚  "          @�\)?u@p�@�  BnQ�B�aH?u@P  @~{BE�HB��                                    By��@  "          @�\)?aG�?���@�Q�B���B�=q?aG�@333@�=qB_=qB�                                      By���  "          @��R>���@	��@�\)B��3B�=q>���@?\)@��BZ�RB���                                    By���  T          @�ff>\?��H@���B���B�{>\@4z�@�33Bc  B��3                                    By�2  T          @��
��G�?�(�@��B�� B��\��G�@&ff@���BmB�                                    By��  "          @�33��R?��@��B�8RB�p���R@�@�Q�By\)Bң�                                    By�#~  "          @�  ��  ?�
=@���B�G�CG���  @
=q@��RB��RB�                                    By�2$  
�          @�(�����?�  @���B��qC�q����@ ��@��B�ǮB�z�                                    By�@�  �          @������?��@�(�B��)CB�����@ff@��\B���B�ff                                    By�Op  
�          @������R?��@��B��qC�\���R@�
@�Q�B��3B�\)                                    By�^  
�          @��\���?�  @�=qB��C����@ ��@���B�B�Q�                                    By�l�  �          @��\���
?�@�G�B���C	�����
@
�H@�
=B|\)B�33                                    By�{b  
�          @�=q�^�R?���@��
B�#�B����^�R@{@���B�aHB޳3                                    By݊  �          @��Ϳ(��?��@�  B��B�33�(��@�@�p�B�k�Bՙ�                                    Byݘ�  
(          @�G��(��?fff@�p�B�B�k��(��?�Q�@�z�B��{Bم                                    ByݧT  
�          @�G��L��?8Q�@�{B��qC
�L��?�\@�{B��B�                                    Byݵ�  �          @�\)�J=q?8Q�@��
B�� C	s3�J=q?�G�@��
B��RB�#�                                    By�Ġ  "          @��H�Q�?!G�@�  B��3C#׿Q�?ٙ�@�Q�B�� B�.                                    By��F  
�          @��R�8Q�>�33@�z�B�  C�8Q�?�z�@��RB�#�B�B�                                    By���  
�          @�  ��
=?�Q�@�33B�� Bڅ��
=@�R@�Q�B�Q�B�#�                                    By��  �          @�{��z�?��@�=qB��B����z�@�@�  B��=Bî                                    By��8  
�          @�녿!G�?��@���B��B홚�!G�@	��@�=qB�B�BԊ=                                    By��  
Z          @��H�(��?n{@��RB�W
B�
=�(��?���@�p�B��B�p�                                    By��  T          @�  �\)?
=q@�{B���C33�\)?�\)@�
=B��B�{                                    By�+*  	�          @��
�5?�\@���B�Q�C=q�5?���@��HB���B�=                                    By�9�  T          @�z�
=q>�@��HB�L�C
uÿ
=q?���@�(�B�#�B�B�                                    By�Hv  �          @��׿�R>��R@�\)B��Cn��R?�z�@���B��\B�                                    By�W  �          @�\)�\)?.{@���B��\C���\)?�G�@���B�z�B��                                    By�e�  "          @�  �\>�Q�@�
=B��Ck��\?�(�@���B��fB���                                    By�th  �          @���s33?p��@�ffB�B�C+��s33?��R@���B���B��)                                    Byރ  
�          @�(��aG�>��@�\)B��fCG��aG�?�p�@���B�B�B��H                                    Byޑ�  T          @�33�fff�u@�ffB�Cc��fff>u@���B�C%+�                                    ByޠZ  "          @�z�@  ����@�
=B�W
Cmٚ�@  =�\)@��HB��
C.�3                                    Byޯ   T          @��ÿ+���33@��B���CtO\�+��W
=@�\)B���CE\)                                    By޽�  �          @�33�G���ff@���B���Co��G����@���B�C>��                                    By��L  �          @�
=�Tzῑ�@�G�B�p�Ci�ÿTz�=L��@���B�.C0�)                                    By���  �          @�Q�8Q쿂�\@�33B��Ck��8Q�<��
@��RB��=C2��                                    By��  �          @�Q�0�׿0��@��B���C`�Ϳ0��>�
=@�{B�L�C�                                    By��>  �          @����Ϳh��@��B�#�Cn����>�=q@���B�ffC^�                                    By��  �          @�(����Ϳ�\)@�
=B���Cz5þ��ͽ#�
@��HB��)C:0�                                    By��  �          @�����aG�@��B��Cr8R��=���@��\B�33C'�q                                    By�$0  �          @��Ϳ   �^�R@���B�Cp{�   =�\)@��
B��HC+��                                    By�2�  �          @��ÿ����=q@�33B�Q�Cp녿����@��B�p�C?ff                                    By�A|  "          @����Tz῀  @��
B�
=Cfn�Tz�=�\)@�
=B���C/h�                                    By�P"  �          @��ͿxQ쿑�@�p�B�#�Ce�R�xQ�L��@���B�B�C70�                                    By�^�  T          @��\��p����H@�{B��C��)��p��Y��@��B���Cv8R                                    By�mn  T          @�
=�\)�@�G�B��C��\)�p��@��
B��fC��R                                    By�|  "          @�zᾔz���@���B�k�C��3��z�0��@��\B��
Cwk�                                    Byߊ�  T          @�33��{��=q@���B��C��R��{�!G�@���B�8RCq��                                    Byߙ`  �          @�(��!G��У�@��HB�Cx޸�!G���
=@�=qB��CU�                                    Byߨ  T          @��&ff��\)@�(�B�Q�Cx8R�&ff��@��B�#�CW(�                                    By߶�  T          @����\�fff@�  B�z�Cpc׿�\>aG�@�=qB�k�C�                                    By��R  �          @��׿aG���  @��HB���Co�׿aG���G�@���B��fCN�
                                    By���  T          @��׿\)���H@���B���CyͿ\)��Q�@�\)B���CT޸                                    By��  
�          @�z�����@��B�\)Cs^���=��
@�33B�  C+:�                                    By��D  �          @��H�W
=�aG�@��B�aHC�]q�W
=>�  @�=qB��{Cn                                    By���  �          @��>.{���
@��\B��HC��>.{?Tz�@���B��HB���                                    By��  "          @��;��J=q@��B�L�Co�׾�>���@�G�B�C��                                    By�6  
�          @��ÿp�׿�=q@�  B�\Cj�H�p�׾k�@�B���CA�                                     By�+�  "          @��Ϳ���
�H@�
=BvffCnE�����ff@��\B�CY�3                                    By�:�  T          @��׿���p�@��HBs(�CoLͿ����{@�\)B��C\33                                    By�I(  "          @�����\)� ��@���B��\Cp�Ϳ�\)�^�R@�33B��CY�                                     By�W�  �          @��H����{@��
B��=Cm�H���333@�p�B��\CR�{                                    By�ft  
�          @��ÿ�G���ff@�z�B��Cf�)��G��Ǯ@��
B�
=CE(�                                    By�u  �          @�ff������@��
B�L�CeǮ���aG�@���B��{C>ٚ                                    By���  "          @�����R�33@�  Bi�HCq�����R��G�@�p�B���CaaH                                    By��f  
�          @��Ϳ��G�@��Bvp�Co�ÿ��xQ�@��B�  C[��                                    By�  
�          @�G����R��G�@�\)B|
=Cj�{���R�:�H@���B���CR�\                                    By௲  "          @���������@�G�BfQ�Ca:���Tz�@�33B��HCL�q                                    By�X  �          @�{��Q����@~�RBh(�Cc{��Q�W
=@�G�B��CNn                                    By���  
�          @�G������=q@�Q�BS�Ch�����Ϳ�33@��RBz�CY:�                                    By�ۤ  �          @�{��
=�AG�@x��BAffCp����
=�G�@�\)Bnp�CfO\                                    By��J  T          @�������P��@��BECx�������@�Q�Bv�Co#�                                    By���  �          @�33���H�<(�@�
=BPp�Cs�����H��{@���B~�Cg�                                    By��  �          @�\)���R�'
=@�Q�B[G�Cp+����R�\@�Q�B��Ca�\                                    By�<  T          @�����=q�%@�Q�BSQ�Cn�H��=q����@�Q�B~��C`�                                    By�$�            @�  ���R�%�@�  BU�Cp  ���R��ff@�  B���Cb+�                                    By�3�  �          @�{��
=�
=@��B_{Cn�
��
=��=q@�Q�B�8RC^�)                                    By�B.  �          @�  ��(��?\)@]p�B;�CwǮ��(���@��\Bm�RCo��                                    By�P�  �          @����z���
@r�\BT(�CjW
��zΎ�@�\)B}  CZ�)                                    By�_z  �          @��ÿ���;�@b�\BBffCzͿ�����R@�z�Bup�Cq��                                    By�n   T          @��=��
���?�\)A�Q�C�� =��
�|(�@��A�  C��\                                    By�|�  T          @��׿���K�@k�B>�CzT{����(�@��HBq�Cr�
                                    By�l  �          @�p��W
=�.�R@h��BO��C|�)�W
=��G�@�ffB�{Ctz�                                    By�  T          @�Q쾔z����?aG�A4  C�R��z����?�A�z�C��\                                    Byᨸ  �          @���<��
��
=�+���Q�C��<��
��
=?!G�@��
C�q                                    By�^  T          @��þ��R��\)�����C�P����R��ff?E�A�\C�N                                    By��  T          @�z�������<#�
=�G�C��q������{?�=qAdz�C���                                    By�Ԫ  �          @ȣ׾��H�Å?�(�A4z�C������H���
@,(�A��C���                                    By��P  �          @׮��Q����@A��\C�ZᾸQ���G�@tz�B
Q�C�!H                                    By���  T          @�z�����\)@G�A��
C�  �����
=@o\)B	�C�޸                                    By� �  �          @љ��u��z�@�RA��C�޸�u����@k�B	33C��
                                    By�B  T          @أ׾�p���=q@��A�C�T{��p�����@x��Bp�C�
                                    By��  �          @أ׾�G��˅@�\A�C����G����H@s33B�\C��)                                    By�,�  "          @�z�����
=@�RA��C�|)�����
=@n{BG�C�#�                                    By�;4  �          @�33�0����(�@z�A��RC��Ϳ0����33@q�B�C�W
                                    By�I�  
�          @θR�0�����@�
A��C��ÿ0�����R@o\)B��C�:�                                    By�X�  �          @Ǯ�G���Q�@��A���C���G���Q�@h��B��C���                                    By�g&  T          @�ff�W
=��
=@�RA�ffC��3�W
=��
=@fffB�C�5�                                    By�u�  T          @�33�@  ���@\)A�z�C�5ÿ@  ���@eBp�C��H                                    By�r  T          @Å�G����R@�\A��
C���G���  @Z�HB=qC��                                    By�  �          @�
=�8Q����?�Q�A�  C�4{�8Q�����@AG�A�
=C���                                    By⡾  �          @��ÿ0������?�p�Aw\)C�P��0�����@1�A��C���                                    By�d  �          @�G�������z�?��RAF{C�Q쾨����(�@(��AٮC�'�                                    By�
  �          @��(����R?�33Adz�C�Ф�(���p�@0��A�\C�xR                                    By�Ͱ  
�          @�
=�+����R?��Az�HC����+���(�@9��A��C�R                                    By��V  T          @���&ff���?��APz�C��׿&ff����@,(�A��C�J=                                    By���  �          @�\)�+���=q?�z�A:ffC��{�+����\@#�
A�ffC�@                                     By���  �          @�33��=q���R?�
=AC�C�����=q���R@#�
A�{C�u�                                    By�H  �          @�
=���
����?�Q�AIp�C�E���
���@!�A�33C�R                                    By��  T          @�=q�fff��?@  A	�C��
�fff���@G�A��C�o\                                    By�%�  �          @�Q�#�
��z�?.{A
�HC��3�#�
���?�ffA�p�C�}q                                    By�4:  T          @�
=>L�����?
=q@�
=C�C�>L�����
?�
=A�\)C�Z�                                    By�B�  �          @�=q>��
���ü��
��\)C�>��
��(�?�An�HC�
                                    By�Q�  
          @��>�Q���녿B�\��C�B�>�Q���33>��@�Q�C�=q                                    By�`,  
�          @�ff>�{����Y����C���>�{���>��@�p�C���                                    By�n�  �          @�z�?������ff��
=C�W
?���33��Q쿎{C�4{                                    By�}x  �          @�Q�#�
��\)��G����C���#�
��{?E�Ap�C��                                    By�  �          @�33��R��  ��ff��{C��{��R���R?333A��C�˅                                    By��  �          @�녿z�H��p�����33C��H�z�H��(�?:�HAQ�C��{                                    By�j  �          @\)��
=�e�>��@�  Cx@ ��
=�Tz�?�A��HCv�q                                    By�  T          @J�H��G���ff>�{AG�C]h���G���33?@  A���CZ�\                                    By�ƶ  �          @?\)�0�׿(��+��Tz�C@\)�0�׿B�\���  CCn                                    By��\  
�          @HQ��@�׿8Q����CAk��@�׿O\)�u���CC#�                                    By��  "          @N{�5����?!G�A6{CMh��5����?xQ�A�p�CI��                                    By��  T          @a��U�O\)?z�A�CA���U��R?G�AN=qC>�
                                    By�N  T          @8Q��.�R>W
=?+�A[�
C/�
�.�R>\?
=A@z�C+��                                    By��  �          @$z��ff?#�
?(�A_�C$���ff?J=q>��AC!��                                    By��  �          @=p����@   ������=qC�R���?��Tz���G�CO\                                    By�-@  �          @C33��G�@"�\���
��
=B�\��G�@
=��G���  B�p�                                    By�;�  
�          @`�׿k�@N{�   ��
B��k�@=p���\)��  B֔{                                    By�J�  T          @a녿�  @N�R��\)��=qB�B���  @A녿�33��33B���                                    By�Y2  �          @j�H�L��@dz�>#�
@!G�B�B��L��@^�R�Tz��QG�B��                                    By�g�  T          @c33���\@1G�?��
A��B�B����\@<��>aG�@~{B��                                    By�v~  T          @z�H�\�h��@J=qBr�CR���\    @R�\B�L�C4\                                    By�$  �          @�z����p�@c�
BW�
Ce�Ϳ��h��@}p�B��qCP�f                                    By��  �          @��ÿ����33@��HBl�HCm&f���ÿTz�@�  B��qCT
                                    By�p  
Z          @�����$z�@n�RBT  Cv.��녿�
=@���B���Cg��                                    By�  T          @������0��@l��BM��CyG������\)@��\B�\Cm�                                    By俼  
Z          @�\)��  �5@hQ�BIp�Cz����  ���H@���B�ffCo�3                                    By��b  �          @�  �J=q�:�H@i��BIffC~Ǯ�J=q���
@�=qB��Cu�                                    By��  
�          @�녿}p��G�@aG�B<\)C|Q�}p��   @�Q�By�Cs�)                                    By��  �          @�������O\)@[�B4  Cz�3�������@�ffBp=qCr�                                    By��T  �          @�33��G��G�@^�RB7�Cx)��G�� ��@�
=BrCn�                                    By��  
�          @��Ϳ����Tz�@Z=qB0�RC{:ῐ���{@��RBm�\Cs\                                    By��  T          @�(��aG��Z�H@W
=B.  C���aG���@�{Bm  Cy@                                     By�&F  T          @�=q�^�R�a�@J�HB$\)C��^�R��R@�G�Bc��Cz��                                    By�4�  "          @��(��u�@B�\Bp�C�|)�(��333@�  BY��C��                                     By�C�  "          @��W
=�n{@`��B*�\C����W
=�#�
@�Bj�C{�f                                    By�R8  �          @��ÿ�ff�^{@p  B5�Cys3��ff�  @��HBq��Co�R                                    By�`�  �          @�ff�\(��o\)@`��B)C����\(��%�@�Bj{C{��                                    By�o�  
�          @��������Q�@Z=qB \)C������7
=@�p�Bb\)C��R                                    By�~*  "          @�G��5�vff@c33B(�HC��)�5�*=q@�Q�Bj=qC~��                                    By��  �          @��.{�qG�@_\)B)(�C��׿.{�'
=@�BjC\)                                    By�v  "          @�����H��ff@`��B��C������H�@  @�=qBaz�C�W
                                    By�  
�          @�G�����
=@b�\B�C��׾��@��@��BbffC���                                    By��  �          @����B�\��  @{�B+{C��)�B�\�:�H@�  Bn�\C�(�                                    By��h  
�          @��׿+��^{@��BH��C��=�+���@��B�G�C|33                                    By��  �          @�\)�������@fffB{C������B�\@�{B^33Cz��                                    By��  T          @�z�=#�
�u@Y��B&33C�O\=#�
�*�H@��
Bj33C�p�                                    By��Z  �          @�=#�
�s33@`  B*�RC�P�=#�
�&ff@��RBn�HC�u�                                    By�   �          @�Q�>\�|��@\��B#�
C�� >\�1G�@�ffBg�C��                                    By��  
�          @���>���  @\(�B!��C�Ff>��3�
@��RBe��C���                                    By�L  �          @�  >������@UBG�C��>���7
=@�(�BbffC�\                                    By�-�  T          @��>�  ��z�@L��BffC���>�  �@��@���B[  C�`                                     By�<�  
�          @��\>\)�|(�@L(�B�
C��>\)�4z�@��RB`�C�h�                                    By�K>  7          @��>Ǯ���H@B�\B  C��
>Ǯ�@  @��
BV��C��\                                    By�Y�  T          @���?@  ���H@A�B�C�+�?@  �@��@�33BS��C��R                                    By�h�  �          @�p�?(���l��@K�B ��C��?(���%@�(�Bd��C�+�                                    By�w0  "          @��R>W
=��Q�@S�
B{C��f>W
=�5@��Bcp�C�&f                                    By��  �          @��?
=q���H@N{B�HC�Ǯ?
=q�<(�@���B\�C�:�                                    By�|  �          @��?@  ���@O\)B33C�+�?@  �<��@�=qB[ffC�!H                                    By�"  "          @�33?fff��33@Tz�B\)C�+�?fff�:=q@���B]
=C��                                    By��  
�          @�G�?�{��(�@B�\B�HC��H?�{�AG�@���BPC�\                                    By��n  �          @�(�?&ff��\)@4z�B�C�e?&ff�K�@}p�BJC��                                    By��  "          @�Q�>��
��G�@<��B�C�C�>��
�=p�@���BV(�C��                                    By�ݺ  "          @�>W
=�}p�@8��B�\C�|)>W
=�9��@}p�BV�HC��                                    By��`  "          @n{>��ٙ�@2�\Bf�C��\>��E�@J�HB�
=C��                                    By��  "          @,��?0��?O\)@�B��qBE?0��?��R@ ��BKG�B�=q                                    By�	�  �          @XQ�>�녿W
=@N{B��\C��{>��>8Q�@Tz�B��)A�p�                                    By�R  �          @�{>�=q��\@X��B_(�C�W
>�=q���@z�HB���C��{                                    By�&�  "          @�33?���@hQ�Bb��C�c�?���{@��B�.C��q                                    By�5�  "          @��H?�=q�z�@�ffBs�RC���?�=q�0��@�(�B�33C���                                    By�DD  "          @�{?B�\�#�
@���Be33C�Ff?B�\��z�@��B�ffC��q                                    By�R�  "          @�Q�?k���R@}p�Ba�C�,�?k����@���B�#�C�k�                                    By�a�  "          @��\��G��h��?�Q�A��HC�  ��G��<(�@.{B+�C��                                    By�p6  T          @����k��\)@
=A�Q�C�b��k��I��@N�RB6�
C��3                                    By�~�  "          @�  ���e�@.{B�\C�{���$z�@l(�B\�C��
                                    By獂  "          @�=�G��l(�@7�B�C�Ф=�G��'�@w�B_��C�&f                                    By�(  �          @�  �#�
�dz�@HQ�B$��C���#�
��H@��\Bm=qC��
                                    By��  
Z          @�(��aG��l(�@I��B!C�G��aG��!�@���Bi��C�~�                                    By�t  T          @�(���p���@s�
Bv(�C��\��p��@  @�Q�B��
Csp�                                    By��  �          @fff����ff@EBo�C������=p�@`  B���C��q                                    By���  "          @��׾\)�ff@~{Bx33C�
=�\)�B�\@�p�B�G�C���                                    By��f  �          @�{�Ǯ���H@x��B��HC|=q�Ǯ=�\)@�=qB���C)�                                    By��  �          @���\)��
=@�=qB���Cx�q�\)�#�
@���B�
=C4��                                    By��  �          @�{�
=q��@���B��C}W
�
=q�\@��
B���CW8R                                    By�X  �          @��R�\�У�@u�B��HC�` �\��{@���B���C]s3                                    By��  �          @mp����׿���@HQ�Bg�\Cj�
���׿
=q@^{B�aHCMh�                                    By�.�  T          @`  ��  �G�@
�HBp�Cq#׿�  ��(�@1�B\�Ce��                                    By�=J  �          @W��J=q��(�@-p�B[�
CuQ�J=q�G�@G
=B��3C`��                                    By�K�  �          @P  �0�׿�
=@#33BY�Cw�3�0�׿J=q@<��B��HCd��                                    By�Z�  "          @qG��\)��Q�@W�B}=qC��{�\)��@n�RB�8RC~��                                    By�i<  �          @���B�\���
@u�B��C����B�\���@�{B�B�Cx�                                    By�w�  �          @�{����R@��\Br��C�g����Q�@��\B�
=CmaH                                    By膈  
�          @`�׿aG���ff@7
=B[�
Cs޸�aG��J=q@Q�B�=qC]�R                                    By�.  �          @S�
���Ϳ�z�@��B@z�Cp)���Ϳ��\@;�B{�C_
=                                    By��  
�          @QG���=q�@�B+
=Cr�Ϳ�=q���@.�RBi��Cf�                                    By�z  �          @e���
=��@G�B �Cm�=��
=��
=@8Q�B[��C`�H                                    By��   T          @�
=��\)�K�@.{B!�C�xR��\)���@fffBlffC�@                                     By���  T          @���=u�Fff@4z�B)
=C��f=u��@j�HBt
=C���                                    By��l  
�          @�Q�>�(��8Q�@G�B<  C�B�>�(���(�@xQ�B��C�
=                                    By��  "          @��?8Q��I��@:=qB(=qC�t{?8Q���\@qG�Bp�RC��q                                    By���  T          @�33?:�H�Dz�@@  B.=qC���?:�H��
=@u�Bv�\C�N                                    By�
^  �          @�
=?���H��@G�B1�C�e?�����H@~{B{  C�~�                                    By�  T          @��R>��R�8Q�@XQ�BE�C��>��R�У�@�z�B�8RC�n                                    By�'�  �          @���?���Vff@33BC���?����H@P��BK�HC��3                                    By�6P  T          @��?�\)�B�\@#33B��C�3?�\)�33@Y��B^ffC�N                                    By�D�  �          @�33?(��/\)@=p�B:(�C�@ ?(��У�@l(�B���C�>�                                    By�S�  
�          @��?Y���)��@K�BC=qC��?Y����(�@w�B��C�
                                    By�bB  �          @�{?���.�R@<(�B3�C�c�?�녿�{@j�HBvC���                                    By�p�  �          @�ff?+��(Q�@L��BF��C�!H?+���Q�@x��B���C�xR                                    By��  "          @�(�?k��-p�@=p�B7�HC�U�?k��˅@l(�B~{C��
                                    By�4  �          @|��?L���$z�@8��B<�C��H?L�Ϳ�p�@dz�B���C�.                                    By��  �          @|(�?\(��   @;�B?��C���?\(����@eB��C��R                                    By髀  T          @}p�?O\)��R@J=qBT=qC���?O\)����@mp�B�W
C���                                    By�&  "          @�Q�?���33@E�BK  C�+�?����33@j�HB�33C�
=                                    By���  �          @��\?����!�@=p�B<33C��H?��Ϳ�z�@hQ�B33C���                                    By��r  �          @�ff?}p��6ff@:=qB0  C���?}p���(�@l(�Bv�C���                                    By��  �          @�  ?�G��:=q@:�HB.  C���?�G����
@n{Bt��C��                                     By���  
�          @�\)?���*=q@E�B;G�C���?�녿��R@r�\B~�C���                                    By�d  �          @�z�?����*=q@>{B7�RC�<)?��Ϳ\@k�B|�C��                                    By�
  T          @��?L���.�R@@��B:�C�'�?L�Ϳ���@o\)B���C�w
                                    By� �  T          @��
?z��)��@FffBCffC�  ?zῺ�H@s�
B��C���                                    By�/V  �          @���?z���R@FffBJG�C���?zῦff@p  B�L�C��q                                    By�=�  T          @�G�>#�
���@O\)BU�C��>#�
��
=@w
=B���C�Ф                                    By�L�  
�          @�ff>\)�!G�@VffBT33C���>\)��  @�  B��C�'�                                    By�[H  T          @�p�    ��R@VffBU��C��    ���H@~�RB�(�C��                                    By�i�  "          @�Q�=�\)�#�
@i��B[z�C���=�\)��Q�@���B��C��3                                    By�x�  �          @��R>��c�
@\(�B/  C��{>����@�B|�C���                                    By�:  �          @��>Ǯ�A�@j�HBH��C���>Ǯ��\)@�
=B�#�C�ٚ                                    By��  
�          @n�R<��
��ff@W
=B��{C�\)<��
���
@k�B���C��
                                    By꤆  �          @U��aG�����@A�B��C��
�aG��#�
@N�RB�B�C>�                                    By�,  �          @R�\��Q�?��
@ ��BQp�B��Ϳ�Q�@�\?���Bz�B���                                    By���  
(          @S33��p�@��?��B(�B�z´p�@1G�?��\A��B�L�                                    By��x  T          @P�׿���@ff?��
BQ�B��)����@3�
?^�RAyG�B�G�                                    By��  
�          @L�Ϳ�33?�@G�B!�\B�LͿ�33@\)?�  A�
=B�q                                    By���  �          @K��fff?G�@7�B���C
��fff?��H@�BM��B�                                    By��j  
Z          @L�;�33��G�@E�B�u�Cg����33?#�
@B�\B�.B�#�                                    By�  T          @2�\���?n{@ ��B���B����?�p�@�\BD
=B��)                                    By��  �          @?\)�\?��@!G�Bt��B��;\@�?�B&�RB�ff                                    By�(\  T          @\(���Q�?@  @U�B��\B��R��Q�?�@8Q�Bep�B���                                    By�7  T          @K���Q�?���@1�Bv��B�Ǯ��Q�@z�@�B(ffBŞ�                                    By�E�  T          @N�R�Ǯ?Tz�@B�\B��B� �Ǯ?���@$z�BX�\B�8R                                    By�TN  "          @N{�c�
?E�@?\)B�#�C:�c�
?޸R@"�\BQ�B���                                    By�b�  �          @P  ���R?W
=@5By  C�῞�R?�G�@Q�B?�B�=q                                    By�q�  
�          @=p����?�@
=BU\)B�� ���@��?�p�B��B�q                                    By�@  "          @7��s33?�  @G�BP  B��s33@(�?�{BffB�                                    By��  �          @1녿�  ?�z�@
�HBMQ�B����  @z�?�ffBp�B瞸                                    By띌  �          @>{���?�
=@ffBS�
B��ÿ��@	��?�(�B\)B�B�                                    By�2  
�          @W
=�s33?�ff@5�Be=qB��f�s33@�@�B\)Bޔ{                                    By��  "          @Z�H�W
=?�G�@<(�Bn�\B�\�W
=@�@\)B$(�B���                                    By��~  �          @[��aG�?�G�@<(�Bmp�B��\�aG�@�@\)B#�\B۽q                                    By��$  T          @W
=�p��?�@0  B\ffB��p��@!G�@   Bp�B��                                    By���  T          @Y����  ?��@,(�BR��B�=��  @'
=?�33BB�                                    By��p  	�          @P  �G�?�(�?�\)B\)C�{�G�@��?�33A�(�C�                                    By�  T          @P���	��?���?��B�Cٚ�	��@�?��HA�G�CO\                                    By��  �          @R�\��\?��R@�
B��C�H��\@�?�z�A�
=C�                                    By�!b  T          @P�׿��H?�\@B!\)C����H@��?��A�Q�B�33                                    By�0  T          @E���Q�?�  ?�G�A�=qC	��Q�@
�H?L��Aw
=C�{                                    By�>�  
�          @C33�޸R?�
=?�p�B%��C�H�޸R@�?�{A��C��                                    By�MT  "          @>{���?�33?˅B�\C
����@ff?h��A�
=C�f                                    By�[�  �          @<�Ϳ�z�?�
=?���A�G�C
����z�@?E�As33C�                                     By�j�  
�          @?\)��(�?��
?���B�C���(�?�=q?�  A�{C	�                                    By�yF  T          @?\)����?p��@�B,�HC\)����?˅?���B �C��                                    By��  "          @=p���(�?�Q�?���BC޸��(�?�  ?��AЏ\C
J=                                    By얒  T          @9������?�
=?��B"(�CY�����?�\?��A��C=q                                    By�8  
�          @;����?G�?��B�HC�����?�\)?�33A�G�Cff                                    By��  
�          @@  ����?k�@��B7�C�ÿ���?�{?�
=B	ffC
�q                                    By�  �          @HQ���?.{?�{B��C$B���?�ff?�  A�z�C��                                    By��*  �          @I���(��>Ǯ?�z�A��C+�
�(��?s33?�A֏\C #�                                    By���  
�          @J�H���>��H?�
=BffC(�����?��?У�A��\C&f                                    By��v  T          @J=q�$z�>�  ?�Q�B�C.n�$z�?Tz�?��RA�\)C"                                    By��  �          @J�H�"�\?5?�Q�B�HC$ff�"�\?�G�?�=qA�G�C�=                                    By��  �          @Dz��(�?J=q?�33B�C!�R�(�?�=q?�G�A�p�Ck�                                    By�h  	�          @G
=�\)>\?�  B  C+:��\)?z�H?�  A��
C��                                    By�)  T          @>�R��?\)@ ��B)��C%
��?��R?�B��C��                                    By�7�  T          @?\)�p�>u?�B33C.p��p�?Q�?�(�A�33C!�
                                    By�FZ  �          @N�R�&ff>aG�?��B�
C/
�&ff?aG�?�Q�A��RC!:�                                    By�U   "          @L(��!G�?��?�=qB=qC'�R�!G�?�?�G�A�z�C0�                                    By�c�  T          @K��?\)@33B:�C$�q�?���?���BG�C=q                                    By�rL  T          @L(���(�?Q�@333By33C쿜(�?�\@z�B<�B�33                                    By��  
�          @Mp���=q?Q�@9��B��HC�\��=q?�ff@=qBC��B��
                                    By폘  
�          @W���p�?+�@<��Bt�Cuÿ�p�?�Q�@ ��BA  C5�                                    By�>  
�          @Vff���?�R@@  B�L�C녿��?�33@%�BJC ��                                    By���  T          @Mp�?�\)?Q�@,(�Bz�
Bp�?�\)?�p�@{B<\)Bd33                                    By���  T          @H�ÿ�{>���@�HB_�C)Q��{?�33@	��B=ffC^�                                    By��0  �          @Dz��G�>�(�@"�\B��=C���G�?���@p�Bgz�B�B�                                    By���  �          @A�?�p�?&ff@!�Be�\A��R?�p�?�G�@�B4Q�B6p�                                    By��|  "          @AG�?�
=>�G�@,(�B��qA�G�?�
=?���@ffBQ(�BE��                                    By��"  "          @G����R?k�@+�Bp=qCs3���R?���@
=qB2  B�k�                                    By��  
�          @U����?���@:=qBy=qC����@33@�B4\)B�                                    By�n  
�          @W��s33>�G�@J=qB�aHC�f�s33?Ǯ@2�\BcQ�B��                                    By�"  "          @`  ��?�@Z�HB��C�H��?�(�@@��BmffB��                                    By�0�  
�          @X�ÿ5>���@QG�B��CE�5?��R@;�Br�B���                                    By�?`  
�          @L��<#�
?+�@C33B��B�<#�
?�p�@&ffBa=qB���                                    By�N  
(          @Vff?��
?O\)@;�Bz��B�H?��
?���@�B>\)B[�R                                    By�\�  
�          @U��G���@I��B�8RC;�
��G�?��\@?\)B��3C�3                                    By�kR  �          @`  �5�#�
@[�B��C7c׿5?��H@Mp�B��)B�                                    By�y�  
�          @S�
�\(�>��@K�B�u�C#Q�\(�?�33@7�Bp��B�B�                                    By  "          @S�
�#�
>�\)@N�RB���C^��#�
?�
=@9��Bv\)B�                                      By�D  "          @]p���=q�u@\(�B��C@� ��=q?���@N{B�W
B�
=                                    By��  "          @_\)��33��33@\(�B��C`�H��33?n{@UB��
Bݳ3                                    By  T          @^{�G���@Q�B��C<s3�G�?��@FffB�33B�L�                                    By��6  
�          @qG������&ff@l��B�B�Cu{����?@  @k�B�8RB�u�                                    By���  �          @w
=�k��G�@qG�B��{C}L;k�?&ff@r�\B�\)Bۮ                                    By���  
�          @tz�\(��&ff@j=qB�{CYB��\(�?:�H@i��B���C�R                                    By��(  �          @qG���  �E�@c33B���CY����  ?z�@e�B�B�C�H                                    By���  
�          @n{���
�:�H@Y��B�8RCQ�Ϳ��
?�@[�B��\C��                                    By�t  T          @b�\��(���Q�@B�\Bo�HC?�)��(�?G�@=p�Be�C��                                    By�  
�          @c33���H�Ǯ@E�BqffC@����H?E�@@��Bh=qC                                    By�)�  �          @g
=��녾���@J�HBw��CA�3���?J=q@FffBn{C=q                                    By�8f  �          @fff��33���@H��BtQ�CF� ��33?#�
@G�Br
=C�3                                    By�G  
(          @c�
�ff�u@7�BWz�C5�{�ff?�  @,(�BD�
Cn                                    By�U�  T          @`  �
�H�L��@.�RBNp�C5B��
�H?u@#�
B<�
C�                                    By�dX  �          @^�R�(���@,(�BK{C7^��(�?^�R@#33B<�
CW
                                    By�r�  
(          @_\)�
=���@1G�BRG�C8��
=?aG�@(Q�BC�HCk�                                    By  "          @e��ff���
@7�BU�
C<� �ff?B�\@2�\BM�C +�                                    By�J  T          @`���ff���R@1G�BR=qC<u��ff?:�H@,(�BIC                                     By��  �          @aG��	����@1G�BP\)C7k��	��?fff@'�BA=qC=q                                    Byﭖ  T          @_\)��p��.{@5�B[��C9���p�?aG�@,(�BL�C�3                                    By�<  
�          @E��{��
=@&ffBe��CBǮ��{?�@$z�Ba�HC �=                                    By���  "          @S33��
=���@3�
Bi��CA�{��
=?(��@0��Bc�\C��                                    By�و  T          @c�
��녾�33@N{B�{CB=q���?^�R@HQ�Byp�C�                                    By��.  
�          @tzῢ�\��{@dz�B���CC���\?�G�@\��B���C�\                                    By���  �          @s�
��  ��(�@^{B��CDͿ�  ?c�
@XQ�B{(�CT{                                    By�z  
Z          @y����녾u@hQ�B���C=�ÿ��?�\)@]p�B{  C
=                                    By�   "          @z=q���þ.{@k�B�{C;(�����?��H@^{BzC	�{                                    By�"�  
�          @y����p��L��@fffB��C;�Ϳ�p�?�33@Z=qBtC
=                                    By�1l  �          @vff�������@g
=B���CA쿥�?��@]p�B�aHC�
                                    By�@  
(          @tzῈ�ÿY��@b�\B���CZ������?�@fffB�u�C�                                    By�N�  T          @n{���Ϳ@  @XQ�B���CP�
����?\)@Z=qB�u�C\)                                    By�]^  T          @r�\��=q�G�@c33B��fCW�Ϳ�=q?��@e�B���C�3                                    By�l  
Z          @y���z�H�   @n�RB�Q�CN�z�H?n{@i��B�k�CY�                                    By�z�  "          @{���녿
=@mp�B���COG����?W
=@j=qB��qC�R                                    By��P  T          @z=q��녿�R@e�B���CL���?B�\@c33B��CE                                    By��  T          @{���G��\)@j�HB��
CK���G�?\(�@g�B�B�C޸                                    By�  T          @z�H��\)�+�@l��B�.CR�q��\)?B�\@k�B��3C��                                    By�B  
�          @x�ÿ���z�@l��B�Q�CQB����?W
=@i��B��qC�3                                    By���  "          @x�ÿ\(��333@n�RB�  C[33�\(�?@  @n{B�
=C
��                                    By�Ҏ  T          @u��0�׿z�@n{B���C\��0��?Y��@j�HB���C5�                                    By��4  
�          @u���{�L��@c33B���CW��{?
=@eB�z�C��                                    By���  T          @p  ��녿E�@G�BoQ�CM:���>�ff@K�Bwz�C$��                                    By���  "          @p���p�����@@  BT\)C>Q��p�?@  @;�BM��C!O\                                    By�&  "          @r�\���H��R@R�\Bt(�CG�H���H?(��@Q�Bs(�C�                                    By��  
(          @mp���=q��Q�@L(�Bn�C?���=q?^�R@EBc  C��                                    By�*r  "          @k���=q��@H��BjCD
��=q?0��@G
=Bg{Cs3                                    By�9  
�          @s�
�����.{@L��BdQ�CG(�����?�@N{Bf�C#�R                                    By�G�  T          @|�Ϳ��R��@g�B��
CG+����R?aG�@c33B��C�                                     By�Vd  
Z          @�33���H�
=@q�B���CJ���H?aG�@n�RB��C\                                    By�e
  
�          @�G���
=�5@xQ�B���CJǮ��
=?L��@w
=B�u�Ck�                                    By�s�  �          @��׿��ÿB�\@q�By��CJ�����?8Q�@r�\Bz��C�\                                    By�V  �          @��
����8Q�@{�BCI�)���?Q�@z=qB}p�Ch�                                    By��  	~          @�{���Ϳp��@��RB�CO  ����?5@�  B�{C�f                                    By�  
<          @�\)���
�k�@�  B}z�CO^����
?&ff@���B��
C޸                                    By�H  	�          @����z�c�
@tz�BtQ�CL����z�?�R@w�Bz=qC"(�                                    By��  "          @�  �   �aG�@h��Bl
=CK�
�   ?��@l��Br��C$�                                    By�˔  "          @�z��  ���@l(�B}ffCF��  ?W
=@h��Bw�HC�                                     By��:  �          @��������H@k�B|�CCB����?n{@eBrffC}q                                    By���  
�          @�p��
�H�L��@\��B`�CH8R�
�H?\)@`  Be\)C%xR                                    By���  T          @�\)�{�W
=@r�\Bg�RCH�{?&ff@u�Bk\)C#��                                    By�,  T          @��\�
�H���\@y��Bj
=CM��
�H?�@�  Bs\)C&!H                                    By��  �          @��\�
�H��G�@y��BiCL���
�H?
=q@\)Br�HC&
=                                    By�#x  �          @�G��
=q���@vffBhQ�CM�H�
=q>��H@|��Br�HC'�                                    By�2  "          @�  �G����@x��Bn��COff�G�?   @�  Bz�C%�                                    By�@�  
�          @�p���
=��=q@u�Bp
=CQ!H��
=>�ff@|��B}p�C&��                                    By�Oj  	�          @��\��׿���@o\)B[Q�CR�
���>.{@~�RBq�C/�{                                    By�^  �          @�Q��,�Ϳ�33@b�\B@��CSp��,�;B�\@y��B\�HC8                                    By�l�  "          @�\)�\)��Q�@o\)BQ��CR��\)=��
@�Q�BhC2�                                    By�{\  "          @�������@uB_�CU(���>\)@�33Bw�
C0T{                                    By�  "          @���
=��Q�@qG�B_ffCV\)�
==�Q�@�G�Byp�C1xR                                    By�  "          @�  �33��(�@mp�B_  CW���33=L��@\)Bz��C2��                                    By�N  �          @}p�������@J�HBV  CW{���L��@\(�Bsp�C5Y�                                    By��  
�          @s�
��\��G�@G�B\�RCWh���\<��
@W
=By
=C3\)                                    By�Ě  
(          @~{�7����@
=B�CI��7��#�
@'�B)ffC7+�                                    By��@  
�          @����E���\)@33BQ�CG��E��#�
@#33Bp�C6�R                                    By���  �          @���U����\@	��A�ffCH޸�U���Q�@{B�C:+�                                    By���  "          @���P�׿��@	��A�=qCK)�P�׾��@!G�B�RC<0�                                    By��2  "          @����Q녿У�@{A��CNs3�Q녿+�@*�HB�\C?xR                                    By��  
�          @�G��8Q��ff@z�Bp�CPQ��8Q���@/\)B,�C>�H                                    By�~  �          @p���ff��ff@,(�B7ffCXn�ff���@E�B\�C?�                                    By�+$  
�          @w
=�����@0  B6  CQ���L��@C�
BQC8�                                    By�9�  
�          @_\)���
��=q@7�Bc�CW8R���
=�Q�@C�
B}C0�=                                    By�Hp  "          @hQ쿮{�J=q@QG�B��CRO\��{?�\@Tz�B���C��                                    By�W  
�          @fff��p��c�
@QG�B�B�CW�῝p�>�
=@W�B��
C!�                                    By�e�  
�          @S�
���
�}p�@1G�Bb�HCT�H���
>\)@;�By�RC.�R                                    By�tb  �          @^�R��\)��G�@A�Br�CXn��\)>aG�@L(�B�
=C*ٚ                                    By�  "          @S�
��녿��@(Q�BU�CT}q���<�@5�Boz�C2�{                                    By�  "          @U��R�xQ�@G�B,33CKz���R��@{B?��C4�\                                    By�T  
�          @^{��ÿJ=q@��B.Q�CF5����>8Q�@ ��B9p�C/�
                                    By��  	�          @[���G����\@5�BdCV���G�>�@@  B|�
C.��                                    By�  
�          @]p���{���@<��Bp  Ce)��{��\)@Mp�B��RC7u�                                    By��F  �          @aG���=q���H@5BfC^xR��=q�#�
@EB��=C5��                                    By���  T          @i����녿k�@>�RB[33CN
=���>�\)@G
=Bi�RC+��                                    By��  "          @l(����Ϳ��@AG�B[
=CQ�q����>.{@L(�Bo(�C.��                                    By��8  �          @tz��녿���@G
=BZ��CR����>\)@S�
Bp��C/�3                                    By��  �          @�
=����Q�@6ffB*�C_Y���׿aG�@\(�B[p�CIh�                                    By��  T          @��
�{�z�@UB8��CbE�{�^�R@~�RBl(�CIn                                    By�$*  T          @������(�@^{B8�HCcG���׿p��@�z�Bm�
CJ�=                                    By�2�  
�          @������{@i��B@��Ce&f��ÿc�
@�=qBw{CJ�3                                    By�Av  
�          @�ff�
=� ��@j=qBN{C_�3�
=��@���BzQ�C@k�                                    By�P  
(          @�
=�(����@mp�BPffC\�)�(����
@�z�Bw�C<c�                                    By�^�  	�          @�{��\��{@�B}(�CYh���\>�G�@��
B��
C&!H                                    By�mh  �          @�33��{��@���Bz�\CYY���{>��@�\)B�\C'��                                    By�|  �          @�z��
=���
@�(�B{�CYLͿ�
=>Ǯ@��B�L�C&��                                    By�  �          @��Ϳ�{����@�{B��)CX����{>��H@��\B��{C#�                                    By��Z  T          @�(����H��\)@��RB�CYs3���H?�@�=qB�{C��                                    By��   �          @��׿�=q���
@��Bz�\C`Ϳ�=q>W
=@�  B�{C,��                                    By���  �          @�33�����  @���B��RCe  ���>B�\@��B���C+��                                    By��L  "          @�Q쿪=q��{@���B�RCf}q��=q>\)@�=qB�W
C.\                                    By���  �          @�Q쿦ff���H@�33B��{Cd(���ff>��R@�=qB�Q�C&��                                    By��  �          @��\��ff����@��HB}CfY���ff=���@�(�B��qC/n                                    By��>  �          @����{��z�@��\ByQ�Cf�3��{<�@��B�ǮC2�                                    By���  �          @�������p�@�33Bt�RCe�ÿ�����@�ffB��)C5J=                                    By��  �          @��ÿ�z���@��Bm\)CcB���zὸQ�@��B�u�C7�                                    By�0  �          @�{��\)��  @�G�Bm��Cc(���\)���
@���B�u�C6Ǯ                                    By�+�  �          @�=q�����
@~�RBp�CgxR����@�33B���C90�                                    By�:|  �          @�G���(��ٙ�@h��Bi33Ce5ÿ�(��B�\@�Q�B��C;0�                                    By�I"  T          @�
=��p��\@o\)By��CgͿ�p�<��
@�G�B�.C2�                                    By�W�  �          @�Q쿥���\)@s�
B~��Cb�f���>8Q�@���B�p�C+޸                                    By�fn  �          @��Ϳ�\)���@|��B�G�C`s3��\)>�=q@��B�  C(�R                                    By�u  �          @����ÿ�{@��\B���Cg�\����>���@�G�B���C$��                                    By���  �          @�33�O\)��  @��HB�Cl���O\)>��@�Q�B��fCk�                                    By��`  �          @�G��B�\��33@~�RB�p�Cq� �B�\>aG�@��RB��RC#}q                                    By��  �          @��\�W
=�
=@�{B�u�CW��W
=?�ff@�33B���C ��                                    By���  �          @�33�W
=�L��@�{B�(�C_�W
=?\(�@�B�
=C5�                                    By��R  �          @��H��z�Tz�@s�
B�  CWc׿�z�?0��@uB�8RCQ�                                    By���  �          @��׿�  �z�H@eB}��CULͿ�  >�@l(�B�\C"�R                                    By�۞  �          @c�
��G���  @@  Bk�CU�{��G�>aG�@J=qB�z�C+�                                    By��D  �          @j�H��Q쿓33@A�B`(�CVO\��Q�=�Q�@O\)By�C0�                                    By���  �          @^{������@\)B9�\CN�����#�
@-p�BO��C5�                                    By��  �          @^�R�  ��z�@�B,�COY��  �.{@(Q�BE��C8c�                                    By�6  �          @K��	����  ?�Q�B�CR.�	����
=@G�B8\)C>�3                                    By�$�  �          @Vff��R����@�B�
CSO\��R��@��B9��C@!H                                    By�3�  �          @�Q���H��ff@U�B`��CP{���H>���@^�RBq{C+^�                                    By�B(  �          @y���
�H���\@>{BF�
CRE�
�H�#�
@N�RB`(�C5(�                                    By�P�  �          @w�����ff@:=qBCCR� ����Q�@L(�B^ffC6h�                                    By�_t  �          @�Q��
=����@C�
BG  CLxR�
=>.{@O\)BWz�C/�
                                    By�n  �          @�G������R@>{B;{CT!H���u@Tz�BYQ�C9�f                                    By�|�  �          @�������   @,��B&�C]u���׿Tz�@P  BU�HCH�                                    By��f  �          @u�
=q��33@-p�B3z�CYT{�
=q�   @HQ�BZ�RC@�                                    By��  �          @z�H�"�\��  @��B��CV���"�\�5@8��B>z�CC��                                    By���  T          @~�R��R���@3�
B3�RCPW
��R�.{@FffBMffC7��                                    By��X  T          @tz���\��Q�@ ��B%�CXh���\�(�@=p�BMQ�CB�                                    By���  �          @j=q��Ϳ˅@(�B'��CW�f��Ϳ��@7
=BNffCA��                                    By�Ԥ  �          @j=q��\����@   B,G�CS)��\����@5BK=qC<0�                                    By��J  �          @_\)����Q�@��B2p�CP�����.{@.{BL��C8^�                                    By���  �          @_\)����  @   B5�CR�����W
=@1�BRz�C9�3                                    By� �  �          @P�׿�����@��B2G�CX�H����
=@'�BX��C@�=                                    By�<  �          @c33�"�\��z�?�Q�A˙�CU(��"�\�}p�?��HB(�CIB�                                    By��  �          @�(��@  ��Q�@!�B=qCT���@  �Tz�@Dz�B2�CC��                                    By�,�  �          @���>{��\@)��B��CR�
�>{�#�
@G�B733C@)                                    By�;.  �          @�p��1녿��@;�B&=qCT޸�1녿��@X��BH��C?5�                                    By�I�  �          @�Q��0�׿��@@  B)�CT�3�0�׿�@\��BK�RC>�H                                    By�Xz  T          @��H�N�R��p�@��BM\)CS�=�N�R�#�
@�z�BgQ�C4�q                                    By�g   �          @�\)�33����@�z�B�
=CT�{�33?�G�@�ffB�\Cٚ                                    By�u�  �          @ə����H����@�Q�B���CSW
���H@ff@��B�  B��                                    By��l  �          @��H�\)���@ə�B�ǮCLn�\)@��@�B���B�p�                                    By��  �          @�ff��R��Q�@��By{CM����R?aG�@���B~ffC ��                                    By���  �          @���S�
��@x��B4�CS�{�S�
���@��
BR��C;!H                                    By��^  �          @��׿�����Q�@�\)B{�
CST{����?
=@��HB�#�C#&f                                    By��  �          @��R������R@�B���CK.����?�Q�@�=qB�G�C�                                    By�ͪ  �          @�녿�{�\(�@���B�CT:΅{?�G�@�  B��RC}q                                    By��P  �          @�33��  �^�R@�33B��CV�f��  ?��\@��\B���C�3                                    By���  �          @�{�xQ�\)@�G�B��3CR=q�xQ�?��
@���B��)B���                                    By���  �          @�{��z῔z�@���B�{CV���z�?:�H@��B�(�CY�                                    By�B  �          @���(��p��@�G�B�p�CP��(�?n{@�G�B���C�)                                    By��  �          @�\)��p����H@���B�L�CW��p�?.{@�z�B�\)C��                                    By�%�  �          @�  ���Ϳ��
@��B�W
CV�q����?��@�(�B���C!��                                    By�44  �          @�=q��
���R@�p�Bu�CW�f��
>\@�z�B��C)�{                                    By�B�  �          @��R��
��@��Bj
=CW�H��
>L��@�{B��RC/                                      By�Q�  �          @���"�\��@�Q�BT��CY&f�"�\�8Q�@�p�Bu��C8
                                    By�`&  �          @�z��4z����@n{B9��CY5��4z�
=@�  B_�C?�                                     By�n�  �          @��
�.{�33@s�
B@�
CY�.{���@���BeG�C=�\                                    By�}r  �          @�33�@  ��p�@g�B4��CU\)�@  ���@�33BU�RC<��                                    By��  �          @���<(���@j�HB5\)CW�q�<(��
=@�{BY�RC?G�                                    By���  �          @��R�N{�   @e�B-\)CSǮ�N{��\@�=qBM  C<�q                                    By��d  �          @�Q��Z�H�@N{B(�CV\)�Z�H�xQ�@w
=B=Q�CC�                                     By��
  �          @����_\)�8Q�@J�HB33C[���_\)��p�@���B:��CK
=                                    By�ư  �          @���e��6ff@B�\BffCZ�=�e���G�@x��B4{CJ�\                                    By��V  T          @����Y���0��@I��B��C[  �Y������@|��B<�CJ                                    By���  �          @�Q��G
=�%@W
=B��C[�G
=���@��HBK�HCH+�                                    By��  �          @�p��/\)�:=q@i��B)��Cb�f�/\)��=q@�
=B^��CM�f                                    By�H  �          @�=q�/\)�:=q@w�B0Q�Cb�q�/\)��  @�p�Bd�HCL��                                    By��  �          @���`���Dz�@p��Bz�C]&f�`�׿�Q�@�(�BJ��CJY�                                    By��  �          @�  �p  �I��@^{B(�C\
=�p  �У�@�z�B<33CK}q                                    By�-:  �          @��H�}p��Vff@L(�A��RC\5��}p���@��RB.�
CM�
                                    By�;�  �          @��R��  �Y��@.�RA��
C\aH��  ���@s33B�HCP#�                                    By�J�  �          @�33�����e@Q�A�z�C[�
�����(�@c33B��CQ��                                    By�Y,  �          @����z��H��@�A�Q�CW�{��z����@FffBp�CN�                                    By�g�  �          @���g
=�Tz�@/\)A�p�C^���g
=��
@qG�B)  CQ�                                     By�vx  T          @���z�H�xQ�@{A���C`���z�H�,(�@n�RB��CVp�                                    By��  �          @�33�j=q��  ?�Q�A��
Ce8R�j=q�Tz�@H��B�
C^(�                                    By���  T          @��H�w��qG�?���Ad��C`E�w��?\)@+�A�CY��                                    By��j  �          @��H���H�R�\@ ��A�Q�CY.���H��
@EB ��CP�                                    By��  �          @�G������7
=@ ��AŮCS�{���Ϳ�p�@X��B	z�CHW
                                    By���  �          @�z������@��@G�A��\CS�{�����33@?\)A���CJ��                                    By��\  �          @��\���\)@�\A�z�CQaH����(�@C33B�CFT{                                    By��  �          @���������@Q�A�CQ8R���ÿ�Q�@7
=B�\CF�\                                    By��  �          @�  ��p��ff@��AظRCN����p�����@?\)B=qCBT{                                    By��N  �          @�(���33��(�@*=qA�
=CI����33��R@G
=B(�C<�                                    By��  �          @������@
=A���CL�\����G�@<(�B��C@ٚ                                    By��  �          @��
��Q���\@"�\A�(�CM�=��Q�u@G
=B�HC@�f                                    By�&@  �          @�  �����33@1G�B   CI������@K�B(�C;.                                    By�4�  �          @�G����
�!G�@�A¸RCQ�����
��  @C33B�CF��                                    By�C�  �          @�G����A�@	��A���CVY�����@G�BffCL��                                    By�R2  �          @�{���\(�?�p�A��CY�\���{@FffA��CQ!H                                    By�`�  
�          @����G��X��?�(�A��\CZT{��G���@E�A��CQ��                                    By�o~  �          @����(��S33?��A���CZ�)��(��Q�@=p�A��HCQ�                                    By�~$  �          @�Q��|(��X��?���A�=qC\�3�|(��\)@<(�B ��CT@                                     By���  �          @�����Q��^�R?��A�G�C\����Q��*�H@,��A�ffCU�f                                    By��p  �          @������\�c�
?�(�AK
=C]����\�7�@(�A�ffCW{                                    By��  �          @���w��w
=?��A,(�C`��w��Mp�@Q�Aʏ\C[��                                    By���  T          @����}p��s33?L��A�C_ٚ�}p��P  @��A�G�C[Y�                                    By��b  �          @�z��s�
�r�\?\)@�  C`���s�
�Tz�?�z�A��C]�                                    By��  �          @����e�w
==�Q�?}p�Cc
�e�c�
?��RA��C`Ǯ                                    By��  
�          @���^{�\)>�(�@��Cd���^{�c�
?�{A��
Ca�3                                    By��T  �          @����c�
�s�
?5@�\)Cb�3�c�
�R�\@33A�33C^�                                     By��  T          @��R�u�����?��A:=qCbk��u��U�@!�A�  C]                                    By��  T          @�  �����_\)?�p�AEG�CZaH�����333@=qA�ffCTxR                                    By�F  T          @�p��g
=�y��?xQ�A$��Cc.�g
=�Q�@z�A�33C^=q                                    By�-�  T          @�(��\)�n{?�ffAV�HC^��\)�?\)@#�
AمCX��                                    By�<�  �          @��\��\)�E�?�
=A�  CX)��\)�
�H@:�HA�Q�CO5�                                    By�K8  �          @�\)��p��S33@	��A�(�CZW
��p��33@L��B��CP�{                                    By�Y�  �          @���u��aG�@��A��C^�\�u���R@U�B\)CT�                                    By�h�  
�          @��r�\�mp�@ ��A�(�C`^��r�\�.�R@N{B
(�CW                                    By�w*  �          @˅���\�xQ�?���AH��CZ� ���\�G�@,(�A�z�CT�{                                    By���  �          @�����\)�y��?�\)A*�HC]���\)�N�R@��A�p�CWǮ                                    By��v  T          @�
=����z�H?W
=AC]�H����Vff@��A��CY=q                                    By��  T          @�\)������?=p�@�=qC]�����c�
@
�HA�G�CY�3                                    By���  �          @���g
=��=q?���A}�Cf��g
=�\��@C33A�
=C_�q                                    By��h  �          @�z��E���H?�\)A��Ch��E�O\)@>�RB�\CbO\                                    By��  �          @�{�Z�H��{?�G�A�(�Cf���Z�H�Q�@I��BQ�C_�q                                    By�ݴ  
�          @��\�HQ����?��
AX��Cj��HQ��c�
@0  A��Cd��                                    By��Z  �          @��\�3�
�n�R@"�\A�z�Ch���3�
�$z�@mp�B0��C^n                                    By��   �          @����E�����@Q�A�  Ch���E��AG�@\(�B\)C`Y�                                    By�	�  �          @�p��u��Q�?���AS�
Cc�q�u�`  @2�\AᙚC^T{                                    By�L  �          @����z�H����?���A_�Cc���z�H�`  @8��A�z�C]�q                                    By�&�  �          @�ff��ff��ff?��
A��C_O\��ff�c33@p�A���CZ��                                    By�5�  �          @�������H>�{@C33Ca�f�����p�?��HA�C^޸                                    By�D>  �          @���������>�  @\)Cak������p�?�{A��HC^޸                                    By�R�  �          @��
��z����>8Q�?�{Cbh���z����?�A��C`
=                                    By�a�  T          @θR��(����=��
?@  CcJ=��(�����?�\A}�Ca(�                                    By�p0  �          @�33������\)        Cc�f������p�?�
=AtQ�Ca��                                    By�~�  �          @Ǯ��\)���Ϳ�33�+
=C^Ǯ��\)��G�>Ǯ@hQ�C_�R                                    By��|  �          @�ff��\)���H��R��  Ca�3��\)��Q�?�ffA�Ca5�                                    By��"  �          @����\)���׿333�ȣ�Cdp���\)���R?��
A(�Cd{                                    By���  �          @�ff��Q���{�!G���\)Cc����Q����
?z�HAffCcxR                                    By��n  �          @�Q��g���{�.{�ٙ�Cf���g�����?n{AQ�Cf�=                                    By��  �          @�33�z=q��  >8Q�?���Ca���z=q�k�?�=qA��C_B�                                    By�ֺ  �          @�����G��y����(���p�C_�R��G��s33?xQ�A{C_@                                     By��`  �          @���qG�����L����
Ccp��qG��z�H?�G�APQ�Cb�                                    By��  �          @����k����H    <#�
CgJ=�k�����?�\)A~�RCen                                    By��  �          @��H�]p����?��
A}G�Cl���]p�����@`  B��Cg�                                    By�R  �          @���`  ���R@4z�A�p�CiT{�`  �Z�H@�G�B$��C`J=                                    By��  
�          @љ���=q��p�?��A�33Ca�H��=q�^�R@S�
A�=qCZٚ                                    By�.�  �          @ʏ\������z�?��Ab{C^k������U@8��A�CXc�                                    By�=D  �          @��������p�?k�A�C^&f����e�@z�A���CY��                                    By�K�  �          @��H��Q��{�?���A&�\C]{��Q��S33@��A�  CX5�                                    By�Z�  �          @�����H�k�?��HA^�\CZ�3���H�<(�@)��A�33CT��                                    By�i6  �          @�����n�R?���Ak33CZ�H����<��@0��A��HCTO\                                    By�w�  �          @ʏ\��  ����?�z�A(��C\B���  �W�@\)A��HCWW
                                    By���  �          @�G������\?�ffAG�C]����]p�@��A�33CX�=                                    By��(  �          @Ǯ���H����?��A?�C](����H�Tz�@'
=AŅCW�
                                    By���  �          @�G���G��qG�?�ffAG33C[� ��G��E@!�A��
CV@                                     By��t  �          @�����\)�k�?���A1G�C[c���\)�C�
@A�z�CVT{                                    By��  �          @�{���R�]p�?��A"{CXL����R�8��@
�HA��
CS��                                    By���  �          @�����H�_\)?xQ�A��CYJ=���H�<(�@�A��CT��                                    By��f  �          @�  ��=q�Z=q?   @�
=CX����=q�A�?��A��CU��                                    By��  �          @\��33�1G�����p�COn��33�,(�?5@أ�CN�                                     By���  �          @�������\(�?Q�@��CVh������<��?���A�{CRn                                    By�
X  �          @����������@,(�A���CaG������HQ�@�  BG�CX@                                     By��  �          @�(�����G�@Q�A�(�C\�����Dz�@X��A�p�CU@                                     By�'�  �          @�ff������G�?�AL(�C\������S�
@.{AǮCV�\                                    By�6J  �          @��\����h��?��HAh(�C\+�����:�H@'
=A�G�CV!H                                    By�D�  �          @�Q���
=�Z=q?�A�{CZ���
=�%�@9��A���CSk�                                    By�S�  �          @�\)��{�Vff?�Q�A���CZ���{�\)@=p�A�(�CR�                                    By�b<  �          @�����\)�W
=@+�A��CZp���\)�  @j=qB�\CO��                                    By�p�  �          @�(����H�P��@,(�AٮCZ}q���H�	��@hQ�B�CO�3                                    By��  �          @������O\)@-p�A�33CYh�����Q�@i��B(�CN�3                                    By��.  �          @��H�����Mp�@0��AᙚCZ�������@l(�Bp�COs3                                    By���  �          @�Q���=q�mp�@.�RAƣ�C[���=q�$z�@tz�B=qCQc�                                    By��z  �          @�=q����mp�@L(�A�\)CZ������H@�Q�B��CO�                                    By��   �          @��
�����%�@c33B33CS��������R@��RB-(�CD
=                                    By���  �          @\���\�333@a�B�CVs3���\����@���B233CG��                                    By��l  �          @ƸR�����G�@Z=qBG�CX�3���Ϳ��@�Q�B-G�CKaH                                    By��  �          @У���z��7
=@w�B��CU(���zῳ33@��B433CE�3                                    By���  �          @أ������E�@k�B��CTٚ���׿�@�  B&�CGJ=                                    By�^  �          @�p���
=�P��@��B�\CX���
=��Q�@�
=B8Q�CH��                                    By�  �          @�{�����|(�@i��A��C[� �����   @�Q�B%�\CO�{                                    By� �  �          @��H��\)���R@VffAمC\(���\)�6ff@��B�
CQ                                    By�/P  
�          @�33��\)���R@C33A��
C]޸��\)�L��@��Bz�CT��                                    By�=�  �          @�33������@7
=A���C_�=����\��@��B{CW�                                    By�L�  �          @�
=���H��33@"�\A��RCb�=���H�~{@��HB�\C[ff                                    By�[B  �          @�\)�����H@5A��HC`u����hQ�@�G�B��CXQ�                                    By�i�  
�          @������R��33@*=qA�{Ca�����R�l��@��
B
(�CZ.                                    By�x�  
�          @�G���ff���@@��A�z�C^� ��ff�Dz�@���B��CU&f                                    By��4  T          @�\)������(�@.{A���C_T{�����O\)@�Q�BQ�CV�f                                    By���  �          @�z���
=��p�@1�A���C]� ��
=�AG�@�  B�CT��                                    By���  �          @�������|��@l(�B�C^�=����!G�@�G�B/33CRT{                                    By��&  T          @�(���{�x��@\)B(�C]5���{��@���B4=qCP
=                                    By���  T          @�\)��33��
=@*�HA�Q�C]
=��33�G�@z=qBp�CT��                                    By��r  
�          @����H�z=q@N{A�\C^����H�)��@�=qB!\)CS^�                                    By��  T          @�
=�����y��@W�A���C^8R�����%@��RB&Q�CS\                                    By���  
�          @�G���33�w�@Tz�A�(�C_L���33�$z�@���B)
=CT�                                    By��d  �          @����w
=�u@VffA���C`�
�w
=�"�\@�p�B.�
CUY�                                    By 
  �          @�ff�g��|(�@eB�\Cch��g��#�
@�p�B:(�CW@                                     By �  �          @ʏ\�n�R�s�
@Z�HB��Ca���n�R�   @�\)B3��CU�\                                    By (V  
�          @˅�tz��qG�@Y��BffC`��tz��{@�{B1=qCT�                                    By 6�  �          @�{�5��=p�@�p�B<CbT{�5���\)@�z�Bj(�CM�3                                    By E�  
�          @\��Q��ff@�Q�Bo�Cfuÿ�Q��@��B�aHCC=q                                    By TH  �          @�G����/\)@�z�B[33Ce)���p��@��B��CI��                                    By b�  T          @ʏ\�0���K�@��RBA  Ce{�0�׿�  @��Bq  CP��                                    By q�  �          @�(��AG��`��@�z�B-�
CeG��AG���@���B_�CTh�                                    By �:  �          @���G��Z�H@���B*Cc���G����@�z�BZ�
CS                                    By ��  �          @�  �Dz��K�@�=qB*�\Cb��Dz��(�@�(�BXCQ0�                                    By ��  �          @�{�E�H��@~�RB(G�Cas3�E���H@���BU�CP�                                    By �,  �          @���HQ��H��@��
B+�RCa��HQ��@��BX�CP�                                    By ��  �          @�=q�E�L(�@��
B+z�Ca��E��(�@�p�BYQ�CQ\                                    By �x  �          @�  �g��A�@�=qB"�HC[���g��˅@�=qBJ�CK�                                    By �  �          @�=q�{��Y��@eB
z�C\ٚ�{��@�\)B4�RCO�q                                    By ��  �          @Ϯ��z��k�@U�A�{C]����z��(�@�=qB'�HCR�=                                    By �j  �          @��p���G�@���B+�C[�R�p�׿��@���BQ��CJ@                                     By  �          @��
�g
=�
=@��HBL\)CU)�g
=���@��Bg\)C<��                                    By�  �          @���hQ��%�@���BGG�CWW
�hQ�J=q@�=qBe��C@@                                     By!\  �          @���k��Fff@�B6��C\+��k���33@��B\�CH�)                                    By0  �          @ٙ��g��|(�@�B  Ccu��g��(�@�ffBH�CU��                                    By>�  �          @ٙ��q���p�@`  A�=qCeu��q��Fff@�ffB/�C[h�                                    ByMN  �          @���s33�c�
@��B�C_!H�s33��@�  BJ�\CP�                                    By[�  �          @ۅ�p  �=p�@��B7
=CZG��p  ���@��\BZz�CF�                                    Byj�  �          @�=q�z�H�Vff@��RB#z�C\}q�z�H��ff@���BK  CL��                                    Byy@  �          @�Q��n{�1G�@�B:�CX��n{��{@�G�B\  CD�{                                    By��  �          @�{�j�H�B�\@�{B2=qC[���j�H��Q�@���BW��CI��                                    By��  �          @�
=�s�
�S�
@�B%�C\��s�
���
@��BL�
CM�                                    By�2  �          @�G��z�H�tz�@�Q�B  C`E�z�H���@�  B=�\CSz�                                    By��  T          @�ff�s�
�]p�@���BC^B��s�
��p�@�(�BH=qCOz�                                    By�~  �          @����k��6ff@��\B2G�CY�q�k���ff@�\)BU(�CG��                                    By�$  �          @\�XQ��\)@��B9�RCXk��XQ쿂�\@��RBZ\)CD�
                                    By��  �          @��Y���%�@���B7�CY@ �Y����\)@�\)BYG�CF.                                    By�p  �          @��
�r�\�>�R@�z�B"CZ5��r�\��=q@��BG33CJ�)                                    By�  �          @����fff�\)@��B;G�CV�R�fff�u@�ffBY��CB��                                    By�  T          @θR�l���#�
@��
B6�HCV��l�Ϳ��@�BU�\CC��                                    Byb  �          @θR�s�
�Mp�@�=qB�
C\!H�s�
��=q@�33BC�CM��                                    By)  �          @�����  �_\)@r�\B33C]
��  ��@�p�B6�
CP�{                                    By7�  �          @�  ����U@n{Bz�CZ�������
@���B1�CNY�                                    ByFT  �          @�G������U@h��BQ�CZ  �����@�\)B,�HCN�                                    ByT�  �          @У�����S33@a�BQ�CY(�����@��B(�CM��                                    Byc�  �          @�  ���\�K�@hQ�B33CXE���\��Q�@��B+�CL.                                    ByrF  �          @�{���R�=p�@w�B�RCW
���R��@��\B5(�CI��                                    By��  �          @�=q���\�B�\@p  Bp�CX�H���\���
@�\)B4�CK�\                                    By��  �          @�(������Dz�@p  B{CXxR���Ϳ�@��B3(�CK�{                                    By�8  �          @�G������S�
@`��B=qCX�R������@��HB&�\CM��                                    By��  �          @У���{�U@Z=qA�z�CX����{��@�  B"��CN
                                    By��  �          @�  ��ff�O\)@\��B �CX{��ff�z�@�Q�B#��CM�                                    By�*  T          @�\)�����H��@Z�HA���CV�
���׿��R@�{B!\)CK�
                                    By��  T          @˅����G
=@W�A��
CW#������p�@�(�B"�CL#�                                    By�v  �          @��
��  �Mp�@I��A�G�CW����  �	��@}p�B��CM}q                                    By�  �          @�Q���p��c�
@4z�A�33CYY���p��%�@o\)B(�CP�                                    By�  �          @�����c�
@#�
A�{CYO\���*=q@_\)B��CQ��                                    Byh  �          @�
=�����c�
@/\)AȸRCYff�����&ff@j=qB	�HCQ:�                                    By"  �          @�Q���\)�c33@.{A��
CX�
��\)�&ff@h��B�
CP�=                                    By0�  �          @Ϯ���H�P��@L��A��HCWff���H�(�@�Q�B  CM�=                                    By?Z  �          @ƸR�����(��@q�B�CT�����Ϳ�Q�@�(�B3��CG�                                    ByN   T          @����p��,��@i��B(�CT����p����
@���B/�\CH33                                    By\�  �          @�  ��ff�<��@c�
B
�
CW���ff��ff@�Q�B+��CK+�                                    BykL  �          @�  ��{�%@uB��CS�f��{��\)@�p�B4(�CF!H                                    Byy�  �          @ƸR��p��'
=@r�\B\)CT���p���z�@�(�B3G�CF�3                                    By��  T          @�G�����)��@n�RB�RCS�\��녿�(�@��HB.\)CF��                                    By�>  �          @�=q���
�6ff@e�B	�HCU����
���H@��B(�\CIY�                                    By��  �          @��������5�@eB�\CU\)������Q�@�  B*\)CIz�                                    By��  �          @ə���=q�6ff@e�B
�\CUp���=q��(�@��B)p�CI��                                    By�0  �          @ə���  �-p�@]p�Bp�CS���  ��\)@��\B"  CG��                                    By��  �          @��
����3�
@]p�Bz�CS�3��녿�(�@��
B CH��                                    By�|  �          @˅�����8��@W�A��CTh���������@���B{CI�{                                    By�"  T          @ə������E�@S33A�ffCW�������@���B
=CL�                                    By��  �          @�Q����H�E@Q�A���CWz����H�33@�Q�B�\CMG�                                    Byn  �          @�Q���p��C�
@Mp�A��\CV�R��p���\@{�B�
CL��                                    By  �          @ə���ff�H��@J=qA��CW+���ff�Q�@z=qB��CM�\                                    By)�  �          @ȣ���(��Tz�@AG�A��CY33��(��ff@tz�B(�CP@                                     By8`  �          @�ff����]p�@4z�A�=qCZ�\����#33@j�HB{CR�{                                    ByG  
�          @˅�����\��@3�
A�ffCYQ������"�\@j=qBCQE                                    ByU�  �          @�33�����b�\@,��A�p�CZ������*=q@dz�B	{CRp�                                    BydR  �          @��H����b�\@0  A�\)CZG�����)��@g�B{CR��                                    Byr�  �          @�z����R�aG�@:=qAأ�CZO\���R�%@p��Bp�CR!H                                    By��  �          @��������g�@A�A�=qC\{�����)��@y��B�CS�f                                    By�D  �          @����s33�QG�@Tz�B��C\�3�s33�\)@��HB+33CR}q                                    By��  �          @�(��\)�]p�@FffA��C\���\)�\)@{�Bp�CS�                                    By��  T          @��
��G��c�
@:=qA���C]k���G��(��@qG�B��CU&f                                    By�6  �          @ƸR��  �l(�@@  A�
=C^�3��  �/\)@x��B  CVff                                    By��  �          @ƸR�����l��@>{A�{C^�{�����0��@w
=BffCVk�                                    Byق  �          @�\)���H�fff@@��A�33C]^����H�*=q@w�B�RCU�                                    By�(  T          @�Q�����dz�@I��A�  C]W
����%@�  B��CT��                                    By��  �          @�  ��(��Tz�@S33A�\)CZ����(��z�@�=qB"��CQE                                    Byt  �          @�������]p�@L(�A��
C[������R@�  B{CR�
                                    By  �          @ʏ\�~�R�i��@S�
A�{C^xR�~�R�(Q�@��B$ffCUp�                                    By"�  �          @ʏ\��  �n�R@L(�A�{C_���  �0  @�=qB�CV��                                    By1f  �          @�(������p��@N�RA�
=C_
�����1G�@��B \)CV��                                    By@  T          @����Q��mp�@VffA�=qC^�R��Q��,(�@��RB$ffCU�\                                    ByN�  �          @�G��}p��e@U�A��C^0��}p��%�@��B%ffCU)                                    By]X  �          @�Q���z��o\)@VffA�  C^)��z��.�R@�
=B!��CU^�                                    Byk�  �          @�\)��
=��z�@(��A��RC`s3��
=�S33@h��B�CZ�                                    Byz�  �          @�\)������33@'
=A�ffC_�R�����QG�@eB�
CY\)                                    By�J  �          @�
=�����  @"�\A�33Ca�����\(�@c�
B��C[�f                                    By��  �          @�Q��������@*�HA�Ca�)�����Y��@k�B	��C[Q�                                    By��  �          @�G�����ff@/\)Aƣ�Ca)���U@n�RB�CZ��                                    By�<  �          @�=q��\)��G�@&ffA�Q�CaaH��\)�^{@g�BC[\)                                    By��  �          @Ӆ�����(�@!G�A�\)C^������U@`  B 
=CX��                                    By҈  �          @�����z���=q@\)A��
C`z���z��a�@`��A�p�CZǮ                                    By�.  �          @Ӆ���
��33@�A�  C`�����
�fff@Y��A�=qC[��                                    By��  �          @Ӆ��33����@ ��A���C`����33�`  @aG�B �HCZ�                                    By�z  �          @�33���\��(�@ffA���C^����\�X��@Tz�A��CX��                                    By   �          @љ�������G�@�HA��C]�)�����R�\@W�A���CW��                                    By�  �          @�Q�����x��@#�
A��HC\z�����G
=@]p�B G�CVY�                                    By*l  �          @�\)���R�x��@'
=A�z�C]����R�Fff@`��Bp�CV�
                                    By9  �          @������R�z=q@.�RA�  C]B����R�Fff@hQ�B�CV�                                    ByG�  �          @�p����R�j�H@0  A�C[�����R�7�@eBffCT�q                                    ByV^  �          @�(����R�\(�@>{Aޣ�CY�
���R�%�@o\)B��CR
=                                    Bye  �          @��
��p��[�@AG�A��HCY�{��p��$z�@q�B  CR+�                                    Bys�  �          @�p�����_\)@7
=AӅCY}q����*�H@h��B
(�CRh�                                    By�P  �          @�p������vff@(��A�ffC]@ �����E�@`��B�HCW�                                    By��  �          @�p���{���\@(Q�A��C`@ ��{�S33@c�
B�RCZB�                                    By��  �          @����(���z�@(��A�Q�Ca���(��W
=@e�B��C[�                                    By�B  �          @�  ����C33@8��Aܣ�CUz�����  @c33B
�RCM�R                                    By��  �          @�������
=@S�
Bz�CO������(�@r�\BG�CE�q                                    Byˎ  �          @������R� ��@C�
A���CP����R��
=@e�B�CG�3                                    By�4  �          @�p���{�!�@G
=A�\CPT{��{��Q�@h��Bp�CGٚ                                    By��  �          @�����R�\)@A�A��HCO�\���R��
=@b�\B33CG��                                    By��  �          @�(������'
=@3�
AڸRCP�H���׿���@W
=B{CI(�                                    By&  �          @������.�R@0��A�{CQ���������R@U�B�RCJxR                                    By�  �          @��������!�@;�A�{CO�����׿�  @\��B�
CH)                                    By#r  T          @�����
���@7
=A�CN�����
��Q�@W�B��CG&f                                    By2  �          @\����{@2�\A�=qCO#���녿�p�@S33BG�CG                                    By@�  �          @\���\�ff@5�A�  CM�����\��{@S33B��CFp�                                    ByOd  �          @�{������@\)A���CK�������{@<��A�G�CEO\                                    By^
  �          @�G���Q��33@*=qA�G�CJ5���Q쿮{@Dz�A�\CC+�                                    Byl�  �          @�p�����1G�@!G�A�ffCQp�����ff@FffA�
=CK&f                                    By{V  �          @������G�@��A���CTL����   @;�Aߙ�CN�)                                    By��  T          @�(���ff�;�@p�A�ffCQp���ff�@5Aә�CL.                                    By��  �          @�p����\�<��@ ��A�ffCP�q���\���@)��A�
=CL@                                     By�H  �          @�����H�333@
�HA�(�CO�����H�{@0��A�{CJ��                                    By��  �          @��
��G��@��?�{A���CQ����G��   @   A�\)CMJ=                                    ByĔ  �          @�p����H�k�?��RA�G�CY=q���H�G�@2�\A��HCT�=                                    By�:  �          @�ff�����n�R?��A�ffCYQ������L��@,��A�(�CU#�                                    By��  T          @Ϯ���z�H?�ffA]��CZ}q���]p�@=qA���CW�                                    By��  �          @�\)�����  ?�z�AH��C[&f����dz�@�\A�33CX                                      By�,  �          @�����^�R@�A��CW^�����8��@:�HA��CR�                                    By�  �          @�(����R�P��@A�Q�CUL����R�(��@AG�A�(�CP                                    Byx  �          @�(���z��Fff@��A�33CS)��z��"�\@2�\A�CN@                                     By+  �          @ə���ff�0��@+�Aʏ\CQ+���ff�@O\)A���CJ�
                                    By9�  �          @�Q������Dz�@�A��RCSc������ ��@0��A�\)CN��                                    ByHj  �          @�  ���
�3�
@�A�{CP�\���
�  @0��A�=qCK�                                     ByW  �          @ȣ������@  @��A�ffCR�������(�@3�
Aՙ�CM��                                    Bye�  �          @�  ��  �>�R@p�A��\CR�\��  �=q@4z�Aי�CM                                    Byt\  �          @�
=��=q�%�@�RA�(�CN�3��=q���H@?\)A�=qCI+�                                    By�  �          @�ff�����@*=qA�G�CL
������@FffA�  CE�=                                    By��  �          @��H���X��@��A�CVu����5�@5A�=qCQ�{                                    By�N  �          @Ǯ��=q�>�R@&ffA���CS� ��=q�@L(�A�(�CM�H                                    By��  �          @�  ��Q��E@$z�A¸RCU  ��Q��p�@L(�A�COE                                    By��  �          @ə����H�H��@ ��A�(�CT����H� ��@H��A�
=COp�                                    By�@  �          @����=q�E@(��A�  CT����=q�(�@P  A�  CN�
                                    By��  �          @����p��N{@.�RA�\)CV����p��#33@W
=B  CP��                                    By�  �          @��H�����`  @
�HA�Q�CX{�����<��@7�A�  CS��                                    By�2  �          @ʏ\�����^�R?�p�A�=qCWc������>{@,(�A���CS=q                                    By	�  �          @�\)���I��@
=A���CT�����'�@/\)A��CO�q                                    By	~  �          @����ff�h��?ǮAj�RCY����ff�N�R@33A�CV}q                                    By	$$  
�          @������g�?���AJ{CY
=�����P  @A�ffCV�                                    By	2�  T          @�
=���R�g�?�=qA�p�CY�����R�I��@#�
A¸RCU                                    By	Ap  �          @�����\)�R�\?�Q�A�Q�CU� ��\)�3�
@&ffA�p�CQp�                                    By	P  �          @��H��  �Z�H?��A�=qCV\)��  �<(�@#�
A�z�CR�                                     By	^�  �          @�33��(��P��?���A�G�CTn��(��333@\)A��\CP��                                    By	mb  �          @ə����H�?\)@��A�p�CRff���H�{@.{A��CM�                                    By	|  �          @�Q���p��?\)@�A���CSL���p���H@@  A�p�CN0�                                    By	��  �          @Ǯ��
=�Fff@��A�=qCS����
=�%@.�RA��
COz�                                    By	�T  �          @�  ��\)�5@{A���CQ����\)���@@��A�ffCLs3                                    By	��  �          @Ǯ�����3�
@ffA�p�CQ0������G�@8Q�A�Q�CL@                                     By	��  �          @ƸR���@��@��A���CSh�����R@5�A���CN�q                                    By	�F  �          @�
=����*�H@�A��CO��������@5�A�(�CJ�R                                    By	��  �          @ƸR��  �
=@"�\A��CIٚ��  ��ff@:�HA�ffCDc�                                    By	�  �          @���p��:=q@A�CR�
��p��Q�@8Q�A�{CM                                    By	�8  T          @�(���33�P��@Q�A���CWQ���33�-p�@?\)A陚CR��                                    By	��  �          @�����p��<(�@+�A���CT(���p��@Mp�A��\CN��                                    By
�  �          @�Q���p��A�@=qA��HCS�
��p���R@=p�A�(�CN�                                    By
*  �          @�\)�����S33@   A�G�CWW
�����.�R@G
=A�p�CRn                                    By
+�  �          @�{��33�X��@�A�CXff��33�7
=@?\)A�p�CS�\                                    By
:v  �          @�=q�����R�\@�A�
=CW�R�����0��@<(�A��
CSY�                                    By
I  �          @����(��QG�@{A£�CX� ��(��.{@Dz�A�(�CS�{                                    By
W�  �          @�{�����Fff@�\A��CVp������%@6ffA�CQ�\                                    By
fh  �          @�����z��;�@   AîCT@ ��z��Q�@AG�A��CO.                                    By
u  �          @�����p��.�R@
�HA���CP�q��p��  @*=qAљ�CL�{                                    By
��  �          @�{��  �<��@'�A���CS����  �Q�@H��A��HCN�H                                    By
�Z  �          @��k���Q�@0��A�G�Cc}q�k��Y��@_\)B\)C^                                    By
�   T          @Ǯ�e���@(Q�A�  Cf���e�qG�@Z�HBQ�CbaH                                    By
��  �          @�p��}p���p�@{A��\Cbu��}p��j�H@>{A��HC^Ǯ                                    By
�L  �          @���u���@Q�A�{CdY��u�u�@:�HA�Q�C`�                                    By
��  �          @�p��\)��z�@��A��Cb\�\)�i��@<��A��HC^k�                                    By
ۘ  �          @�{����x��@�A��HC^�����X��@>�RA��
CZ��                                    By
�>  �          @Ǯ��Q�����@A��RC_}q��Q��e�@4z�A��HC[�q                                    By
��  �          @�����p��z�H@A�{C]�\��p��]p�@333AӮCZ                                    By�  �          @�G���\)�fff@�A���CY8R��\)�I��@*�HA���CU��                                    By0  �          @ƸR����Z=q@
=A���CX5�����:=q@<��A��CT                                      By$�  �          @����z��9��@*=qAָRCUs3��z��ff@I��B\)CP5�                                    By3|  �          @������,��@6ffA�Q�CT
����Q�@S�
B

=CNG�                                    ByB"  �          @�=q��Q��h��@��A��\C^:���Q��K�@1G�A�CZp�                                    ByP�  �          @�(�����J=q@#�
AθRCX�R����(��@FffA��\CS�f                                    By_n  
�          @��R�����A�@)��A���CV}q�����\)@J=qA�CQ}q                                    Byn  �          @�����  �=p�@1G�A�p�CU\)��  �=q@P��BG�CP(�                                    By|�  �          @�����G��333@1�A�z�CS�R��G��  @P  B��CNh�                                    By�`  �          @\��
=�0  @)��A���CRB���
=��R@FffA��
CMJ=                                    By�  �          @�����33�.{@5�A�G�CR����33�
�H@QG�B�CME                                    By��  �          @�=q����.�R@1G�AمCRff����(�@N{B 
=CM5�                                    By�R  �          @�z���{�5@1G�A�=qCS+���{�33@N�RA�CN�                                    By��  �          @\���R�+�@.�RA�p�CQ�����R�
=q@J=qA��HCL��                                    ByԞ  �          @�z�����+�@4z�A�p�CQ� ������@P  B 33CLJ=                                    By�D  �          @�p���G��!�@=p�A�CO�{��G���(�@W
=B{CJT{                                    By��  �          @ƸR�����%�@AG�A�
=CP\)����� ��@[�B
=CJ�\                                    By �  �          @���=q� ��@;�A��HCOz���=q���H@U�B\)CJ
                                    By6  �          @��������H@E�A�RCO������@]p�B��CIB�                                    By�  
�          @�33��z����@FffA�\CO����z���@^�RB�CJ�                                    By,�  �          @�������(��@7�A��CQ�����
=@Q�B �CK�H                                    By;(  �          @�=q��  �\)@3�
A�G�CO�H��  ��(�@L��A���CJ�                                     ByI�  �          @����=q��@.�RA�z�CN����=q��
=@FffA��HCIǮ                                    ByXt  �          @�����{��@B�\A�(�CM�f��{��p�@XQ�Bz�CH5�                                    Byg  �          @��R���H�Q�@=p�A�=qCOY����H��@Tz�BffCI�
                                    Byu�  �          @��R��G��G�@HQ�A��CN����G����H@^{B(�CH��                                    By�f  �          @�p������
=@W
=B
{CP�����׿�\@l��B�HCJxR                                    By�  �          @�Q����R�(�@l(�Bz�CO� ���R���@�Q�B'{CH#�                                    By��  �          @�G����Q�@;�A�p�CLz�����{@P  B��CG                                    By�X  �          @�\)��33�.{@�
A��\CP!H��33�33@.�RA�Q�CLQ�                                    By��  �          @Å��=q�'
=@(�A�CO5���=q�p�@%A�ffCK�\                                    Byͤ  �          @�����ff�{@��A��CN���ff��\@1G�A��HCJn                                    By�J  �          @��
�����!�@ffA�
=CN�������
=@.�RA�z�CJ�3                                    By��  �          @����=q�   @�A��CNB���=q�z�@333A���CJ=q                                    By��  �          @�
=���
�%@�HA���CN�{���
�
=q@333A�=qCJ�                                    By<  �          @�{��=q� ��@ ��A�
=CNQ���=q�z�@8Q�Aݙ�CJ:�                                    By�  �          @�\)����#33@%A�ffCN������ff@=p�A�G�CJ�=                                    By%�  �          @�
=��ff��R@z�A�\)CM����ff��@,(�A��HCI�=                                    By4.  �          @�
=������@p�A�Q�CJ�{������@"�\A���CGJ=                                    ByB�  �          @��
��(���@z�A�=qCL�=��(���(�@*=qAΣ�CI�                                    ByQz  �          @�(����\��\?�(�A��
CK=q���\����@33A���CH�                                    By`   �          @�(������{?�z�A�p�CM)�������@G�A�z�CJ�                                    Byn�  �          @��������"�\?��A��\CM�������p�@  A�{CJ�                                     By}l  �          @������
=@G�A��CL�����G�@�A�
=CH�
                                    By�  �          @�Q���{�ff@1G�Aޣ�CN����{���@FffA�  CI�q                                    By��  �          @�������  @>�RA�Q�COW
�����\@S33B  CJ�                                    By�^  �          @�(���Q���H@p�AŮCN�3��Q��G�@333A�CJ�                                    By�  �          @�  ��Q���H@>�RA�  CQ�\��Q��Q�@S�
B  CLk�                                    Byƪ  �          @��H��(��=q@>�RA��RCP޸��(���
=@Tz�B
�HCK��                                    By�P  �          @��������
@.{A�CN����녿��@B�\A���CJ\)                                    By��  �          @�p������\)@;�A�
=CO���������
@N�RB(�CJp�                                    By�  �          @��
��G��@  A�(�COE��G���p�@$z�A�CK��                                    ByB  �          @���33�Q�@A���COY���33�   @*�HA�=qCK��                                    By�  �          @���p��ff@
=A���CP��p�� ��@�A�
=CL}q                                    By�  �          @�33��\)���H@�A�  CJ�\��\)��{@"�\A�  CF�                                    By-4  �          @��
���R���@
=qA���CM����R����@p�AϮCI��                                    By;�  T          @����(��\)@\)A�=qCM�=��(����@"�\Aי�CJ(�                                    ByJ�  �          @�G����R���@p�A�Q�CM�{���R�G�@1�AۅCJ#�                                    ByY&  �          @�ff��
=�=q@z�A��CL�q��
=�33@(��A��
CIc�                                    Byg�  �          @Å��p���@  A��RCLE��p���p�@#�
A�Q�CH�q                                    Byvr  �          @�=q�����\@�RA���CK������Q�@"�\AŮCH��                                    By�  �          @�G����R�p�@ffA��HCJ�3���R���@��A��HCG�H                                    By��  �          @��H�����{@A���CJ�
���׿�33@��A���CG�{                                    By�d  �          @�G�����(�@z�A�Q�CJ��������@
=A��CG�q                                    By�
  �          @�����ff�z�@�
A��HCL���ff�   @
=A�CI
                                    By��  �          @�{��G��
�H@G�A��CKG���G���=q@#33AˮCG�3                                    By�V  �          @�{��Q���R@�A�z�CK���Q���@$z�A���CH��                                    By��  �          @�z����R��R@��A�Q�CL@ ���R��33@"�\A��HCH��                                    By�  �          @��������\@�A�  CI���������H@"�\A�  CF��                                    By�H  
�          @��
���H���@�A�p�CGT{���H���H@ ��Aʏ\CC��                                    By�  �          @��H���\�ٙ�@z�A�\)CF�=���\��\)@!�A͙�CC�                                    By�  �          @��R��ff���H@�A���CH�
��ff��33@
=A�G�CE�
                                    By&:  �          @����Ϳ�@�\A��CG����Ϳ�G�@!G�A�33CDc�                                    By4�  �          @��R����
�H@��A�
=CK������{@�HA�CH                                    ByC�  �          @��R��
=�(Q�?���A�G�CO���
=�ff@�\A��HCMJ=                                    ByR,  �          @������
�!G�?�Q�A��CN8R���
�\)@��A�CK��                                    By`�  �          @����33� ��?�z�A�Q�CN+���33��R@{A�  CK�H                                    Byox  �          @�����(��.�R?�
=A�
=CP���(��\)@�A�z�CMٚ                                    By~  �          @�����G��4z�?޸RA�{CQ0���G��#�
@A�z�CN�3                                    By��  �          @�
=����6ff?���A��CR(�����%@
�HA�
=CO��                                    By�j  �          @������׿���@
=A��RCE����׿��
@#33A�33CBO\                                    By�  T          @�����녿�p�@�A���CIk���녿��H@��A�  CF�H                                    By��  �          @�p���
=��
?��A�
=CJ����
=��@�\A��\CH\                                    By�\  �          @�(���{�z�?�=qA��HCI�R��{����@�A��CGJ=                                    By�  �          @����R��?��A���CJ�q���R��Q�@33A�(�CHff                                    By�  �          @��\��z��{?�\)A�
=CKT{��z���R?��A��CI33                                    By�N  �          @�����
=���?�  AG\)CJ�)��
=�G�?\Ar�HCI5�                                    By�  �          @�=q��Q��Q�?�  Ao33CH33��Q��p�?�p�A���CF:�                                    By�  �          @��\���H��33?���AU��CG�����H��(�?���AzffCE�\                                    By@  �          @�
=�����z�?�{A\  CI�H���Ϳ��?���A�z�CH
                                    By-�  �          @�33���R���?�G�Ap��CJJ=���R��
=?�\A�
=CHY�                                    By<�  �          @��\������?�G�Apz�CK�R�����
?��
A�(�CI��                                    ByK2  �          @�G������  ?�(�A���CL\�����G�?��RA��HCI޸                                    ByY�  
�          @�33����p�?��A�
CL:�����   ?�ffA�  CJ8R                                    Byh~  �          @������33?�(�Av=qCJ�\����{?��HA���CH�H                                    Byw$  �          @�ff���H�?�\)Aip�CKff���H��z�?�{A��CI�{                                    By��  �          @�=q�����Q�?�p�Aw33CKu����Ϳ�Q�?�(�A��RCI�=                                    By�p  �          @�ff�����   ?�\A��RCI�����ÿ�\?��RA�  CGff                                    By�  �          @�p����R�   ?�=qA�G�CI�����R��\@�
A���CG��                                    By��  �          @��������\)?0��@�p�CR{�������?z�HA4(�CQ
                                    By�b  �          @��H��=q�$z�?^�RAz�CR����=q���?�z�AR�\CQ�                                     By�  �          @�z���
=�z�?�  A3�CO^���
=��?�G�Ac�CN�                                    Byݮ  �          @�p���z��   ?��A>=qCKY���z��\)?��Ag33CI��                                    By�T  �          @������H��\?�\)A�33CH)���H����?�A��\CF�                                    By��  �          @�
=���ٙ�?�{A��HCG\����G�?��A�33CE�                                    By	�  T          @�  ���\���?�A��CI5����\��z�@ ��A�CF�                                    ByF  �          @�{���H���@ffA���CC{���H���@�RA�  C@T{                                    By&�  �          @�  ���Ϳ�@�
A��CD#����Ϳ�
=@��A���CA�                                     By5�  �          @�
=��33��\)?��HA��CFxR��33���@Q�A�
=CC��                                    ByD8  �          @�\)��
=��33@�RA��
CEs3��
=��33@�A��CBc�                                    ByR�  �          @�G���33����@�A��CE�=��33���H@G�A�(�CB�R                                    Bya�  �          @���Q��!G�?!G�@�p�CO�)��Q���?fffA��CO�                                    Byp*  �          @����{�'
=?5@�G�CQ{��{�!G�?}p�A)G�CP0�                                    By~�  �          @��H��33��Q�?�\)A�{CJ����33��(�@z�A��RCH�                                     By�v  �          @�\)���ÿ�{?�Q�A�CI8R���ÿУ�@Q�A���CF޸                                    By�  �          @�(���z��  @A��HCE\��z῞�R@\)A�Q�CB33                                    By��  �          @�G���G���ff@z�Ař�CE���G����@{A�CC�                                    By�h  �          @�G���p��У�@��A�p�CHJ=��p���\)@#33A�G�CE33                                    By�  �          @��\��ff�޸R@ffA��CIY���ff��p�@!�A�\CF^�                                    Byִ  T          @��
��ff��Q�@�A�\)CK�=��ff��
=@{AۮCH�                                     By�Z  B          @�(����Ϳ��H@=qA�z�CK�����Ϳ�Q�@&ffA�
=CI                                    By�   �          @�z���ff�\)@!�A��
CP)��ff���H@0  A�p�CM{                                    By�  �          @����33��G�@�HA�=qCG&f��33��  @$z�A�RCD                                    ByL  �          @��R���׿\@!�A�p�CG�����׿�  @+�A�=qCD\)                                    By�  �          @�33��녿���@=qA�ffCC�\��녿s33@!G�A��
C@^�                                    By.�  �          @�(����R���@8Q�B�RC:O\���R�.{@:=qB
z�C6E                                    By=>  �          @�p����\���@'�A�\)CA�R���\�G�@-p�A��C>E                                    ByK�  �          @�����G����@�
AܸRCA����G��Q�@=qA�RC>                                    ByZ�  �          @�Q���\)����?޸RA�33CE���\)��Q�?�\)A�  CB�f                                    Byi0  �          @�����{��p�?��A��
CE}q��{���?�
=A��HCCaH                                    Byw�  T          @�ff��
=��33@�A�{CG�R��
=��
=@��A���CE�{                                    By�|  T          @�  ��ff���?���A��\CJ���ff��
@
�HA�\)CH�f                                    By�"  �          @Ǯ��
=�	��@$z�A��
CJc���
=���@1G�A�=qCG�)                                    By��  �          @�p���
=�{@{A���CLJ=��
=�\)@��A�z�CJ@                                     By�n  �          @�ff��
=��H@�A��CK޸��
=��@&ffA��\CI�3                                    By�  �          @����  �.�R@*�HA��CNn��  �p�@:�HA�
=CL)                                    ByϺ  �          Arff�C���  @�G�A���CI:��C���=q@�RA�  CFh�                                    By�`  �          A\)�]����
@ʏ\A�(�CEz��]��o\)@�
=Aȏ\CC{                                    By�  T          Av�\�PQ����\@���A�Q�CJG��PQ����R@�Q�A�(�CG�H                                    By��  �          A�z��UG���z�@�p�A��HCM���UG���@��A���CK{                                    By
R  �          A��H�E��=q@��RA�ffCTL��E�ᙚA�RA���CQ�3                                    By�  �          A���8z��	��@�=qA��
CX���8z����
A��A�
=CVO\                                    By'�  �          A�
=�J�H�{@�Q�A�p�CUxR�J�H���@��AЏ\CSc�                                    By6D  �          A�ff�X����z�@�=qA�33CN{�X���Å@��A�p�CLB�                                    ByD�  T          A�{�P  ��{@\A��CP�3�P  ��33@�
=A��CN�f                                    ByS�  �          A����`�����H@���A�ffCM��`����{@�  A�=qCKǮ                                    Byb6  �          A���o�����@�z�A�p�CGn�o���p�@�33A��RCET{                                    Byp�  �          A���a���  @أ�A�
=CG�\�a����@�{A�Q�CE8R                                    By�  �          A|���]G�����@���A�33C<�]G�����@�A�{C9xR                                    By�(  �          Az�H�b�\?�G�@���A���C/��b�\?�{@�G�A�33C-��                                    By��  �          Aw��]��h��@�  A��
C7�]�����@ٙ�AѮC5=q                                    By�t  �          Ax  �\�׿�\)@׮A�p�C9���\�׿B�\@��HAң�C7#�                                    By�  �          Av�H�^�R��@��HA�z�C6.�^�R=u@ӅA�33C3                                    By��  �          Aup��]�����@�\)A�=qC9s3�]��=p�@�=qA�G�C7�                                    By�f  �          Ay��`(��\@�z�A�C:5��`(��n{@�  A�G�C7��                                    By�  �          Apz��Vff��\)@��
A�z�C;���Vff���@�Q�A�33C9}q                                    By��  �          A�  �x���=q@�=qA�p�C<�\�x�ÿ޸R@�  AҸRC:c�                                    ByX  T          A���xQ��0��@���A�33C>��xQ��
=@�A�\)C;�                                     By�  �          A��R�x���0��@�  A�G�C>{�x���Q�@�RA�p�C;��                                    By �  �          Ap(��^ff���
@�
=A���C5Q��^ff>.{@�\)A��C3L�                                    By/J  �          Ao�
�_
=?+�@�ffA���C1:��_
=?��@�(�A�(�C/W
                                    By=�  �          Al(��]��=���@���A�C3�
�]��?�@�Q�A��HC1�{                                    ByL�  �          Aq���[��xQ�@�ffA�
=C8��[����@�Q�A��C5�3                                    By[<  �          A|z��b=q�,(�@��HA�p�C>��b=q�
�H@�G�A��C<�R                                    Byi�  �          A�{�o���G�@�33A�ffC9�o���  @�ffA���C7�\                                    Byx�  �          A�{�s33@!�@�As33C*�\�s33@7�@}p�AeC)Q�                                    By�.  �          A}p��iG�?ٙ�@�=qA�z�C-T{�iG�@�@�p�A���C+�q                                    By��  T          A}�c�@N�R@�{A���C'5��c�@h��@��A�  C%�H                                    By�z  �          Ar�H�^=q@fff@�G�A�C%xR�^=q@~�R@�\)A��C$                                    By�   �          Ak
=�Y��@L��@��A�=qC&� �Y��@c33@�z�A�G�C%^�                                    By��  T          Ag33�Zff?��H@��A�=qC,�H�Zff@�
@�Q�A�33C+n                                    By�l  �          Aj�H�\��?:�H@��
A�\)C0�R�\��?�\)@���A��HC/\)                                    By�  �          Ah  �Y�?#�
@��HA�z�C1Q��Y�?��\@���A�=qC/�3                                    By��  �          Ag33�X��>\@�33A��C2c��X��?E�@��A�(�C0�                                    By�^  �          Aj�\�[
=�Ǯ@��
A�{C5�f�[
=<#�
@�z�A��\C3�3                                    By  �          Ak\)�\Q�xQ�@�ffA�C8��\Q��@�Q�A�C6aH                                    By�  �          Ak��`  �W
=@�p�A�=qC4ٚ�`  >\)@�A�Q�C3n                                    By(P  �          Ap���d�׿Tz�@�G�A��RC7O\�d�׾��@��HA�Q�C5�H                                    By6�  �          Au��i����@��A��C8u��i��B�\@��A�  C7                                      ByE�  �          Au�j�R��G�@�=qA��HC7��j�R�+�@�(�A���C6�H                                    ByTB  �          Aw�
�fff���@�Q�A�33C;p��fff��(�@�z�A�33C9�
                                    Byb�  T          Ayp��c\)�Z=q@��A�ffCA���c\)�?\)@���A�{C?��                                    Byq�  �          Av=q�[
=���H@���A�{CE�
�[
=�z=q@��\A�{CC�                                    By�4  �          Av�\�XQ���33@�A�p�CE�
�XQ��x��@�  A���CD�                                    By��  �          A~�R�`z���\)@�ffA�{CE�R�`z�����@ȣ�A�{CD                                      By��  �          A��
�i����  @陚Aȣ�CH���i����@�{A�(�CF��                                    By�&  h          A�=q�]������A (�A�ffCJ�R�]����{A�\A�
=CH�=                                    By��  �          A�G��\����A\)A�\)CM�\�\������A
=ACK��                                    By�r  �          A����Yp��׮@��A�
=CNaH�Yp���z�AG�A�CLT{                                    By�  �          A�G��Yp����H@��RAՙ�CO�\�Yp���Q�A\)A��CM�
                                    By�  �          A�ff�]���p�@�
=A�CN���]����HA33A�(�CL�{                                    By�d  �          A�G��c
=��\)@��A�\)CN.�c
=��z�Ap�A�CL=q                                    By
  �          A���`(����@�ffA���CMh��`(���=qA�\A��CKh�                                    By�  T          A����`����{@��
A��
CL�)�`�����A��A�
=CJ��                                    By!V  �          A��
�]���ffAp�A�  CM޸�]����HA��A��
CK�                                    By/�  �          A���]���=qA��A�
=CM\)�]���
=A�
A�z�CKG�                                    By>�  
�          AxQ��IG����@��A��CPǮ�IG���\)@ʏ\A���CO=q                                    ByMH  "          At(��@(���{@�33A�ffCR��@(��׮@��HẠ�CQT{                                    By[�  �          Aw\)�H�����@33@��CS���H����=q@%A   CS
                                    Byj�  
�          Av�H�N{�
=�^�R�O\)CU@ �N{����\)��ffCU^�                                    Byy:  T          Av{�G���\?�  @o\)CW�=�G���?�\)@���CWB�                                    By��  
�          Ai��=G���{@s33Aq�CU�=G����@�=qA��CT\                                    By��  T          Aw��J�R��z�@�\)A�=qCQff�J�R��Q�@��RA�p�CP
                                    By�,  "          A����W\)�θR@��
A�Q�CM�f�W\)����@љ�A��CL!H                                    By��  �          A{��=q�:�H@�ffA�=qCk!H�=q�3�@�\)AծCj�                                    By�x  
�          A�ff�G��7\)@�{A�Cfٚ�G��0Q�@�ffAͮCe��                                    By�  
�          A{��8���Q�@��HA��\CXff�8����@���AУ�CV�                                    By��  4          Ap���Lz���\)@θRA�  CGL��Lz�����@�  A�CE��                                    By�j  
Z          Av�R�_\)��ff@�
=A��C:Q��_\)��{@��A���C8��                                    By�  T          A����X���#33A�HA��C>���X�׿��RAG�A��\C<aH                                    By�  T          A����S��Z�HA(�A�33CB}q�S��6ffA�B�C@&f                                    By\  "          A��
�[
=���AA�(�C>!H�[
=���A  A��HC;ٚ                                    By)  
�          A�33�Y��
=A{A���C=�)�Y���ffAQ�A�33C;�\                                    By7�  T          A�p��[\)�6ffA�A�C?�[\)�z�A�
A�G�C=�)                                    ByFN  �          A
=�]��
=@��A�z�C<���]녿�\)@�A�z�C:�f                                    ByT�  T          A~�R�\  ��@�A���C=�H�\  ��@��A�G�C;��                                    Byc�  �          A�33�F�H��=qA�A��
CN�3�F�H����A(�A���CL�3                                    Byr@  �          A�\)�8(��=q@�(�A�CWB��8(���A{A�=qCU}q                                    By��  �          A�ff�<������@�\)A���CU�{�<�����
A�A��RCT                                    By��  �          A�ff�=G�� Q�@�=qA噚CV!H�=G���  A��A���CTc�                                    By�2  T          A���9p��A�\A�CZ.�9p��	G�A�A�\)CX�                                     By��  �          A�33�0�����A33A�\C]�=�0���  A��A��C[�                                    By�~  �          A����1���A�A�ffC\���1�(�A��A��HC[
=                                    By�$  �          A�z��'��p�@�=qA�
=C]�3�'��G�A=qA��
C\)                                    By��  T          A�33�$(��=qAffA���C_:��$(����A�
B��C]��                                    By�p  "          A���#33��AG�A��RC_�)�#33��HA�RB
=C]�q                                    By�  
�          A�  ������AQ�B��Ca������Ap�B�
C_8R                                    By�  T          A��p��
=A��B��C_�q�p���A%G�B33C]��                                    Byb  
�          A�(������RA�B

=Cb&f����G�A��B
=C`^�                                    By"  
H          A������&�\A��B�HCd@ ����p�A�RB33Cb��                                    By0�  
l          A�33�Q��$(�A(�B=qCc��Q���HA�Bz�Cb=q                                    By?T  
Z          A�p���R��AQ�B��Ca+���R��RABQ�C_�                                     ByM�  	�          A�=q�����RA��B�
C`�)������A�RBp�C^�H                                    By\�  T          A}G����-�A
�RB�RCj�����$z�A��B�HCi=q                                    BykF  
z          Ax����  �0��A\)A���Cl�3��  �(z�A��B
�
Ckn                                    Byy�  
�          Az=q��G��+�A��B {Cj  ��G��#\)A�\B{Ch�f                                    By��  4          Ay����R�%Ap�A��Cf����R�A
=B\)Ce�                                    By�8  B          AyG��ff�#�A\)B��Cgp��ff�33A��B33Ce��                                    By��  �          Av{�����#�AG�B=qChO\�����\)A�RB�
Cf��                                    By��  "          Ap��������A33Bz�Ch��������Az�B�Cg�                                    By�*  �          Arff�����4  @�Q�A��\Coٚ�����,Q�A�\B�HCn�                                    By��  "          Au��Ǯ�<z�@�=qA�RCr
�Ǯ�5�A  B(�Cq!H                                    By�v  T          At�����H�<��@�\)A�
=Cp�{���H�5@���A�Q�Co��                                    By�  
�          Ar�R�Å�M�@���A���Ct���Å�G�
@�  A���Cs�                                    By��            Aq�����*�R@ָRAң�Cg�����$Q�@�=qA�\)Cf�\                                    Byh  
�          As��
{�4��@�A�\)Ch�)�
{�/\)@\A�Q�CgǮ                                    By  
�          Ap����\)�B{@�G�Ay�Cl����\)�=�@�\)A�z�Cl)                                    By)�  T          Ar{��Q��@Q�@�=qA���Cn�)��Q��:�H@�  A�\)Cn!H                                    By8Z  �          As33�Ӆ�B=q@���A��\Cqh��Ӆ�<Q�@ָRAӅCp��                                    ByG   �          A{�
��Q��;�
A\)A���Cp�q��Q��4(�A�B	33Co�3                                    ByU�  �          A|  �θR�<  A(�A��Cq+��θR�4(�A�\B	�Cp!H                                    BydL  
�          Axz���p��=�@�ffA�z�Crs3��p��5��A	B�RCq}q                                    Byr�  
�          As33����:{@�33A��\CsB�����2�RA�
B��CrO\                                    By��  T          Ao�
�����3�
A (�B\)Cr������,Q�A
{B��Cq��                                    By�>  
�          As
=��33�A�@���A��Cx!H��33�9�A33Bz�Cw\)                                    By��  
�          A{\)��\)�8z�A=qB�Cq����\)�0��AQ�BCp��                                    By��  
�          Ax����p��(��A{BQ�CkǮ��p�� ��A33B�
Cjp�                                    By�0  �          Azff���)p�A�B  Ck�f���!�A��Bz�CjL�                                    By��  T          Atz����
� z�Ap�B�
Cj�����
�(�A{B�Ci+�                                    By�|  t          Aq����33���Az�B(�Ci=q��33���A��B=qCg��                                    By�"  �          As\)��=q��\A�B�
Ce����=q�
=qA�B\)Cc�
                                    By��  "          AxQ���  �!p�A�B33Ci^���  ��AB=qCg�                                    Byn  
�          Ayp��أ��!��A�RB
=Cl.�أ����A#\)B$ffCj��                                    By  T          Ap����ff�33ABCf
��ff�
=A��BQ�Cdu�                                    By"�  
�          Aq��z���A
=B��Ce����z��33A�RB$=qCc��                                    By1`  B          A{���(��#
=A�B�RCj���(���\A�RB��Ch�f                                    By@  4          A�  ���3�AQ�B{Cl�����+
=A!BffCk5�                                    ByN�  �          A�����{�K\)A��BffCrB���{�B�HA#�B\)Cq:�                                    By]R  
�          A�����ff�@��A$  Bz�Cqٚ��ff�7�A.=qB\)Cp�                                    Byk�  4          A����\�H  A��B�\Cw�H���\�?33A(Q�B  Cv�R                                    Byz�  �          A��\���H�C�
AQ�B=qCt�����H�;\)A"�\B\)Cs�                                     By�D            A�  ��G��?�
A   B\)Cv33��G��7
=A*{B ��Cu+�                                    By��  �          A�p�����K\)A{B	33Cw�
����C
=A ��B�\Cv��                                    By��  
�          A�  ��\)�H��A(�B
ffCuk���\)�@(�A"�RB�\Ctz�                                    By�6            A�ff��=q�J{Ap�Bz�Cx���=q�Ap�A((�B�
Cw:�                                    By��  �          A�z�����Mp�A�B	z�CwǮ����D��A"�\B�
Cv�                                    By҂  T          A�  �����J{A�\B��CwL������A��A%�B{Cvh�                                    By�(  
�          A�{��Q��EG�A\)B	�Cs
��Q��<��A!BffCr{                                    By��  
�          A�33��{�D(�A
=B�HCrE��{�;�A%G�B��Cq33                                    By�t  
(          A����p��aA&�RB�HCt�R��p��X��A2�\B�HCs�=                                    By  �          A�����F�\A.�RB��Cp�����=�A9�B Q�CoJ=                                    By�  T          A�\)�����NffA(��B�HCr�q�����EG�A3\)B��Cq��                                    By*f  �          A�{��p��E�A6ffB{Cr�{��p��<  A@z�B'��Cq\)                                    By9  �          A����  �?�
A;\)B ��Co����  �5��AEG�B+  CnT{                                    ByG�  �          A�Q���(��G33A:�\B=qCq�3��(��=�AD��B(�HCp��                                    ByVX  �          A������G33A0��BQ�CoG������=��A;
=B�Cn�                                    Byd�            A�p����?33A-�B��Ck+����5A7�
B��Ci��                                    Bys�  �          A��\�	��6�\A/�
B33Ch��	��,��A9G�BCgp�                                    By�J  �          A�����8��A&ffB��Ch.���/�
A/�
B{Cf�=                                    By��  �          A�=q� z���RA3�
BG�C`�� z����A<  B"��C^�H                                    By��  �          A��\�9���33A!B(�C[+��9����\A)p�B��CY�                                    By�<  �          A�ff�O���p�A�HB��CSaH�O�����A!p�B{CQ�                                    By��  
�          A�
=�0���*�\A(�A�  C_���0���"ffA ��B=qC^��                                    Byˈ  �          A�{� ���Dz�A$  B��Cl�� ���;�A.{B��Ck��                                    By�.  �          A�������:�\A0(�B{Cjc�����0��A9BCh��                                    By��  �          A����\�<(�A+
=B
=Ck=q��\�2�HA4��B�
Ci�H                                    By�z  �          A���� Q��<  A,z�B��Ck��� Q��2�RA6{Bp�CjT{                                    By   �          A�
=�  �<��A*�\B{Ck��  �3�A4(�B�
Ci�                                    By�  T          A������R�BffA6=qBz�Cm�)���R�8��A@(�B#�ClB�                                    By#l  T          A�{�  �<Q�A1��BffCh�{�  �2�RA;33B�
Cg!H                                    By2  �          A�Q��=q�;�A1G�B�HCh��=q�1�A:�HB=qCf��                                    By@�  �          A�  �.=q� z�A%��B�HC^���.=q��A-B�HC]�                                    ByO^  �          A�(��C�
�G�A��B �
CX���C�
���A ��B�HCV��                                    By^  �          A����Ip����A�A�33CV�q�Ip����A�RB\)CUh�                                    Byl�  �          A�G��P�����A�
A��CT���P�����HA�\BG�CS�                                    By{P  
�          A�
=�]p���Ap�A���CP  �]p���33A\)BG�CNT{                                    By��  �          A��Z�\��=qA�HB33CM��Z�\����A (�B=qCK�f                                    By��  �          A����`(����A$  B  CJ�3�`(���=qA(��Bp�CHǮ                                    By�B  �          A��H��G��X  A-�B33Cq�H��G��N�RA8��B�RCp��                                    By��  �          A�(����IA/�B\)Cl)���@Q�A9�B=qCj��                                    ByĎ  �          A�����G�A/33B\)Cl.��>{A9G�BG�Cj�H                                    By�4  �          A�z�����VffA&=qB
G�Cr������Mp�A1�B�HCq��                                    By��  �          A����ڏ\�T��A&ffB\)Cr�\�ڏ\�K�A1G�B  CqǮ                                    By��  �          A������W�
A%p�B	�Cr�����N�RA0z�BCq��                                    By�&  �          A�����[�A$(�B�CsG�����R�RA/\)B��CrQ�                                    By�  �          A�33��ff�X��A$��BG�Cr�H��ff�P  A0(�B��Cq��                                    Byr  �          A�  ��G��Xz�A'
=B	��Cr� ��G��O\)A2=qB=qCqz�                                    By+  �          A�{��{�S�A,Q�Bp�Cqs3��{�J{A733B��CpY�                                    By9�  
�          A����z��V{A-��B=qCq�f��z��L��A8z�B�
Cp��                                    ByHd  �          A�����XQ�A*�HB33Cq������O
=A6{BCp��                                    ByW
  �          A�����z��W
=A(Q�B	��Cq.��z��M�A3�B\)Cp)                                    Bye�  �          A��R����[�A!G�BQ�Cs������R�HA,��B�Cr��                                    BytV  �          A���˅�]p�A33B�\CuT{�˅�T��A*�\B�Cts3                                    By��  �          A��H��33�ZffA ��B��Cu���33�Q��A,(�B�Ct&f                                    By��  �          A��R��
=�[
=A�\B�CwW
��
=�R=qA)�B�Cv�                                     By�H  �          A�ff�����Z�RA z�B	=qCxp������Q�A+�
B�Cw��                                    By��  �          A�  ����V�RA!p�B
�Cvٚ����MA,��B�
Cu��                                    By��  �          A�p���33�Xz�A"ffBQ�CyY���33�O�A-B�RCx�=                                    By�:  �          A�Q�����X(�A ��B��Cz�����O\)A,  B=qCy:�                                    By��  �          A�p���33�QA z�B�Cv�H��33�H��A+�BCu��                                    By�  �          A�������P��A�
B�Cv�R����G�A*�RBCu�\                                    By�,  
�          A��\��\)�QAz�B	ffCvc���\)�I�A'\)B��Cu}q                                    By�  �          A�����p��R�\A ��B�Cu���p��IA,  B�Ct�{                                    Byx  �          A����=q�^ffA=qA�ffCy
=��=q�V=qA{B	�CxQ�                                    By$  �          A�33���b�RA(�A�Cy���Z�\A(�B\)CxO\                                    By2�  �          A��
���[33A�HBz�Cw}q���R�\A&ffB�
Cv��                                    ByAj  
�          A��
��ff�UA z�B	Cv  ��ff�L��A+�
B
=Cu{                                    ByP  �          A����%��c
=A��B  C��R�%��Z�HA��B33C���                                    By^�  �          A�������mp�AA�(�C�Q쿱��e�AffBz�C�:�                                    Bym\  �          A����
=q�l��A��A�=qC����
=q�d��AG�B p�C��3                                    By|  �          A����(��p��@��Aә�C�5ÿ�(��i��A�A�ffC�!H                                    By��  �          A��H�����p��@޸RA�ffC�n�����j�\@���A�p�C�j=                                    By�N  T          A���?���s
=@�ffA�G�C���?���mG�@��A�=qC��                                    By��  
�          A�ff?h���pQ�@�
=A��C��)?h���jff@���A̸RC��f                                    By��  �          A�=q�����L  A�B��Cw�\�����C33A&ffB33Cv��                                    By�@  "          A����\�0(�A)�B��Ciu���\�&�\A2�\B ��Cg�                                    By��  T          A�{�^�R��Q�A1p�Bp�CG�=�^�R��z�A5��Bz�CE�                                     By�  
�          A�ff�[�
���
A&�HB
�CL�[�
��G�A,  B�HCI�3                                    By�2  �          A�p��[���33A!�BCJ.�[�����A%BQ�CH!H                                    By��  "          A�33�L(����A#
=B�
CL���L(�����A(  B��CJ�                                     By~  T          A��H�O�����A)B�HCK(��O���ffA.�\B�CH�H                                    By$  
�          A�������A<(�B1�C^� �����{AC33B:�C\\                                    By+�  �          A�G��
=� ��A*�\B\)Ce�R�
=�
=A3\)B&
=Cd0�                                    By:p  T          A�������ffA ��Bp�Cg�������A)G�B$z�CfJ=                                    ByI  	�          Av�\�6ff�>�\A�B��C�G��6ff�6�RA�B\)C�H                                    ByW�  
�          Ap��?�=q�IAG�BG�C�p�?�=q�B{Az�B(�C���                                    Byfb  
�          Apz�?\(��P��@�33A���C��?\(��IAG�B\)C��3                                    Byu  T          Ao�
�#�
�H��A�\B�C����#�
�@��AB(�C���                                    By��  �          Aq�>���H��A�B  C�Y�>���A�AQ�B{C�]q                                    By�T  t          Ao�>u�L(�@��A�C���>u�D��Az�B  C���                                    By��  
          Aj�\?
=�H(�@�A�ffC�W
?
=�@��A�B
Q�C�c�                                    By��  T          Ai��?s33�IG�@�33A��C�(�?s33�B=qA�B{C�=q                                    By�F  �          Ak33?c�
�O�@ۅAޏ\C��R?c�
�H��@�33A���C�
=                                    By��  �          An�H?L���S�
@ڏ\A�  C���?L���M�@�\A�z�C�Ǯ                                    Byے  T          Avff?�Q��X��@�Aޏ\C���?�Q��Q�@��RA�
=C���                                    By�8  t          Ao�?Q��W33@�{A�ffC��q?Q��P��@�
=A��C��=                                    By��  �          Ak33>W
=�[�@�\)A��RC�o\>W
=�Vff@���A���C�q�                                    By �  �          Ap  ?�G��O�
@���A��C��f?�G��H��A z�B��C��                                     By *  T          Ak�?���N�H@�p�A�ffC��R?���H  @�p�A�G�C��                                    By $�  �          Am�?�
=�R=q@ۅA��C��?�
=�K\)@�(�A��HC���                                    By 3v  �          Ak
=?}p��P(�@׮Aڣ�C�.?}p��Ip�@�  A��C�AH                                    By B            Ag�
?�Q��N�\@�ffAӅC��?�Q��H(�@�ffA��C���                                    By P�  f          Aj�H@z��Vff@�33A���C�e@z��P��@�z�AΣ�C���                                    By _h  �          Ap��@\)�\Q�@��\A�z�C�� @\)�V�\@���Aə�C��                                     By n  �          Ao�@!��\  @�(�A�z�C�8R@!��Vff@�ffAÅC�Z�                                    By |�  �          Ao�@&ff�]p�@��\A��RC�T{@&ff�X(�@��A��
C�t{                                    By �Z  
�          An�\@{�\  @�=qA�33C��R@{�Vff@�z�A£�C��
                                    