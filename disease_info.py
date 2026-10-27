# disease_info.py — Basis Pengetahuan Agronomi: Gejala & Penanganan Penyakit Tanaman
# ---------------------------------------------------------------------------------
import re

DISEASE_DB = {
    "anthracnose": {
        "nama_id": "Antraknosa (Patek / Cacar Daun)",
        "penyebab": "Jamur patogen (Colletotrichum gloeosporioides / Colletotrichum spp.)",
        "gejala": "Bercak melingkar cokelat gelap hingga kehitaman dengan tepi tegas dan pola cincin konsentris. Pada kondisi lembap, terdapat titik spora oranye/merah muda di tengah bercak.",
        "penanganan": [
            "Pangkas dan bakar daun atau buah yang menunjukkan gejala bercak parah.",
            "Hindari menyiram dari atas kanopi (overhead watering) agar spora jamur tidak memercik ke daun lain.",
            "Semprotkan fungisida kontak berbahan aktif Mankozeb atau Propineb sebagai pencegahan.",
            "Jika serangan meluas, gunakan fungisida sistemik berbahan aktif Azoksistrobin atau Difenokonazol."
        ]
    },
    "bacterialwilt": {
        "nama_id": "Layu Bakteri (Bacterial Wilt)",
        "penyebab": "Bakteri tular-tanah (Ralstonia solanacearum)",
        "gejala": "Daun layu mendadak di siang hari terik tetapi tampak segar kembali pada malam/pagi hari. Warna daun tetap hijau saat layu (tidak menguning). Pembuluh batang berubah kecokelatan.",
        "penanganan": [
            "Segera cabut tanaman yang layu permanen dan musnahkan (jangan dikomposkan).",
            "Taburkan kapur pertanian (Dolomit) untuk menaikkan pH tanah menjadi netral (pH 6.5 - 7.0).",
            "Aplikasi agens hayati bakteri antagonis seperti Pseudomonas fluorescens atau Bacillus subtilis ke tanah.",
            "Kocorkan bakterisida berbahan aktif Tembaga Hidroksida atau Oksitetrasiklin pada lubang tanam di sekitarnya."
        ]
    },
    "blackrot": {
        "nama_id": "Busuk Hitam (Black Rot)",
        "penyebab": "Bakteri (Xanthomonas campestris pv. campestris)",
        "gejala": "Bercak nekrotik kuning berbentuk huruf 'V' khas yang meluas dari tepi daun ke arah tulang daun utama. Tulang dan urat daun menghitam.",
        "penanganan": [
            "Gunakan benih yang sudah diberi perlakuan air panas (hot water treatment).",
            "Lakukan rotasi tanaman dengan famili selain kubis-kubisan (Brassicaceae) minimal 2-3 tahun.",
            "Semprotkan bakterisida berbahan aktif Tembaga Oksiklorida untuk menekan penyebaran pada daun sehat."
        ]
    },
    "blight": {
        "nama_id": "Hawar Daun (Leaf Blight)",
        "penyebab": "Jamur patogen (Phytophthora infestans / Alternaria spp.)",
        "gejala": "Bercak basah keabu-abuan atau cokelat gelap tak beraturan yang meluas sangat cepat. Daun tampak seperti tersiram air panas lalu mongering dan rapuh.",
        "penanganan": [
            "Pangkas daun-daun bagian bawah yang dekat dengan permukaan tanah.",
            "Gunakan mulsa plastik hitam perak untuk mencegah percikan air tanah ke daun.",
            "Semprotkan fungisida sistemik berbahan aktif Simoksanil, Dimetomorf, atau Metalaksil saat cuaca lembap/hujan."
        ]
    },
    "downymildew": {
        "nama_id": "Embun Bulu / Embun Tepung Palsu (Downy Mildew)",
        "penyebab": "Oomycetes jamur semu (Pseudoperonospora cubensis / Peronospora spp.)",
        "gejala": "Bercak bersudut kuning kehijauan di permukaan atas daun yang dibatasi oleh urat daun. Pada permukaan bawah daun, muncul lapisan kapang halus berwarna kelabu keunguan saat udara lembap.",
        "penanganan": [
            "Atur jarak tanam lebih renggang untuk memperlancar sirkulasi udara dan mengurangi kelembapan kanopi.",
            "Bersihkan gulma di sekitar bedengan.",
            "Aplikasikan fungisida spesifik Oomycetes seperti Simoksanil + Mankozeb atau Mandipropamid."
        ]
    },
    "cercospora": {
        "nama_id": "Bercak Daun Cercospora (Frogeye Leaf Spot)",
        "penyebab": "Jamur (Cercospora melongenae / Cercospora spp.)",
        "gejala": "Bercak bulat kecil berwarna cokelat dengan bagian tengah berwarna abu-abu terang menyerupai mata katak. Daun yang terserang parah akan menguning dan rontok sebelum waktunya.",
        "penanganan": [
            "Sanitasi lingkungan kebun dengan membuang sisa tanaman yang terinfeksi.",
            "Semprotkan fungisida berbahan aktif Klorotalonil, Mankozeb, atau Karbendazim sesuai dosis anjuran."
        ]
    },
    "begomovirus": {
        "nama_id": "Virus Kuning Begomovirus / Gemini",
        "penyebab": "Begomovirus (Ditularkan oleh vektor Kutu Kebul / Bemisia tabaci)",
        "gejala": "Urat daun menguning jernih (vein clearing), helai daun menguning cerah, melengkung ke atas, menebal, kaku, dan tanaman menjadi kerdil tidak berbuah normal.",
        "penanganan": [
            "Kendalikan vektor Kutu Kebul dengan memasang perangkap likat kuning (yellow sticky trap) 40 buah/hektar.",
            "Semprot insektisida sistemik seperti Imidakloprid, Tiametoksal, atau Abamektin berseling.",
            "Cabut tanaman yang terinfeksi parah sejak dini agar tidak menjadi sumber inokulum bagi tanaman lain."
        ]
    },
    "verticillium": {
        "nama_id": "Layu Verticillium (Verticillium Wilt)",
        "penyebab": "Jamur tular-tanah (Verticillium dahliae)",
        "gejala": "Daun menguning berbentuk huruf V di tepi daun, biasanya dimulai dari daun bagian bawah pada satu sisi tanaman. Tanaman kerdil dan layu perlahan.",
        "penanganan": [
            "Gunakan media tanam atau tanah yang sudah disterilisasi atau diberi Trichoderma sp.",
            "Hindari luka mekanis pada akar saat penyiangan atau pemupukan.",
            "Terapkan rotasi tanaman dengan tanaman bukan inang (seperti jagung atau padi)."
        ]
    },
    "fusarium": {
        "nama_id": "Layu Fusarium (Fusarium Wilt)",
        "penyebab": "Jamur tular-tanah (Fusarium oxysporum)",
        "gejala": "Tulang daun pucat, daun menguning dari bawah ke atas, tanaman layu perlahan terutama saat tanah kering. Jika batang dipotong melintang, berkas pembuluh tampak cincin cokelat gelap.",
        "penanganan": [
            "Perbaiki aerasi dan drainase tanah agar tidak ada air tergenang di sekitar perakaran.",
            "Kocorkan agen antagonis Trichoderma harzianum sejak awal tanam (fase persemaian).",
            "Gunakan varietas tahan dan pupuk dengan perbandingan Kalium (K) yang cukup tinggi untuk memperkuat dinding sel tanaman."
        ]
    },
    "leafrot": {
        "nama_id": "Busuk Daun / Busuk Basah (Leaf Rot)",
        "penyebab": "Jamur Rhizoctonia solani atau Bakteri Pectobacterium carotovorum",
        "gejala": "Daun membusuk basah, berlendir, lembek, dan sering mengeluarkan bau busuk menyengat. Jaringan daun cepat hancur jika diremas.",
        "penanganan": [
            "Kurangi frekuensi penyiraman dan jangan menyiram daun saat matahari sudah terbenam.",
            "Potong daun yang mulai busuk dengan gunting steril dan buang jauh dari kebun.",
            "Semprotkan fungisida/bakterisida tembaga untuk mengeringkan luka pada jaringan tanaman."
        ]
    },
    "leafspot": {
        "nama_id": "Bercak Daun Umum (Leaf Spot / Alternaria)",
        "penyebab": "Jamur patogen daun (Alternaria spp. / Septoria spp.)",
        "gejala": "Bercak cokelat melingkar hingga bersudut pada helai daun dengan batas kuning tipis di tepinya. Daun mengering mulai dari ujung.",
        "penanganan": [
            "Jaga kebersihan bedengan dan buang daun-daun tua di pangkal batang.",
            "Aplikasi fungisida kontak seperti Propineb atau Mankozeb setiap 7-10 hari pada musim hujan."
        ]
    },
    "mosaic": {
        "nama_id": "Penyakit Virus Mosaik (Mosaic Virus)",
        "penyebab": "Tobacco/Cucumber/Turnip Mosaic Virus (Ditularkan kutu daun Aphid & getah)",
        "gejala": "Warna daun belang-belang hijau muda dan hijau tua berselang-seling seperti pola mosaik. Permukaan daun berkerut, melepuh (*blistering*), dan mengecil.",
        "penanganan": [
            "Sterilkan tangan dan alat pangkas sebelum menyentuh tanaman sehat.",
            "Kendalikan kutu daun (Aphids) menggunakan insektisida nabati tembakau/mimba atau bahan aktif Tiametoksam.",
            "Eradikasi (cabut dan musnahkan) tanaman yang menunjukkan gejala mosaik berat."
        ]
    },
    "fleabeetle": {
        "nama_id": "Serangan Kumbang Daun / Kutu Loncat (Flea Beetle)",
        "penyebab": "Hama kumbang daun kecil pelompat (Phyllotreta spp.)",
        "gejala": "Terdapat banyak lubang-lubang kecil berukuran 1-2 mm pada permukaan daun menyerupai bekas tembakan senapan angin (shot-hole pattern). Pertumbuhan bibit terhambat.",
        "penanganan": [
            "Gunakan perangkap perekat berwarna kuning di sekitar bedengan.",
            "Taburkan abu kayu atau tanah diatom di sekitar pangkal tanaman untuk menghalau kumbang.",
            "Semprotkan insektisida kontak berbahan aktif Sipermetrin atau Deltametrin pada pagi/sore hari saat hama aktif."
        ]
    },
    "rust": {
        "nama_id": "Penyakit Karat Daun (Rust)",
        "penyebab": "Jamur obligat (Puccinia spp. / Uromyces spp.)",
        "gejala": "Muncul bintil-bintil kecil (pustul) yang pecah mengeluarkan serbuk berwarna cokelat kemerahan atau jingga menyerupai karat besi di bagian bawah daun.",
        "penanganan": [
            "Segera buang daun yang memiliki pustul karat agar serbuk spora tidak tertiup angin ke tanaman lain.",
            "Hindari pemberian pupuk Nitrogen (Urea) berlebihan yang membuat daun terlalu rimbun dan sukulen.",
            "Semprotkan fungisida golongan triazol seperti Tebukonazol atau Heksakonazol."
        ]
    },
    "septoria": {
        "nama_id": "Bercak Daun Septoria (Septoria Leaf Spot)",
        "penyebab": "Jamur (Septoria lycopersici)",
        "gejala": "Bercak-bercak melingkar kecil (diameter 2-3 mm) dengan tepi berwarna cokelat tua dan bagian tengah abu-abu keputihan berbintik hitam kecil.",
        "penanganan": [
            "Pangkas daun bagian bawah hingga setinggi 20 cm dari tanah untuk memutus rantai percikan spora.",
            "Terapkan mulsa jerami atau plastik di atas tanah bedengan.",
            "Semprot fungisida Klorotalonil atau Difenokonazol secara berkala."
        ]
    },
    "tomatobacterialspot": {
        "nama_id": "Bercak Bakteri Tomat (Bacterial Spot)",
        "penyebab": "Bakteri (Xanthomonas perforans / Xanthomonas vesicatoria)",
        "gejala": "Bercak basah kecil cokelat kehitaman berukuran 1-3 mm, sering dikelilingi lingkaran halo berwarna kuning terang. Daun menguning dan rontok.",
        "penanganan": [
            "Hindari bekerja di lahan saat tanaman masih basah oleh embun pagi.",
            "Gunakan bakterisida Tembaga Oksiklorida yang dikombinasikan dengan fungisida Mankozeb.",
            "Pastikan benih berasal dari sumber bersertifikat bebas bakteri."
        ]
    },
    "leafcurl": {
        "nama_id": "Virus Keriting Daun (Leaf Curl Virus)",
        "penyebab": "Tomato Leaf Curl Virus (ToLCV) ditularkan oleh kutu kebul",
        "gejala": "Tepi daun melengkung ke atas atau ke bawah secara ekstrem, daun kaku dan rapuh, ukuran daun mengecil abnormal, ruas cabang memendek.",
        "penanganan": [
            "Gunakan mulsa perak untuk memantulkan sinar matahari yang tidak disukai serangga vektor.",
            "Semprot insektisida penekan vektor seperti Piriproksifen atau Spiromesifen.",
            "Beri pupuk daun dengan unsur mikro (Zinc, Boron) dan asam amino untuk merangsang pemulihan pucuk baru."
        ]
    },
    "spottedwilt": {
        "nama_id": "Virus Layu Berbintik (Tomato Spotted Wilt Virus / TSWV)",
        "penyebab": "Tospovirus (Ditularkan oleh hama Trips / Thrips tabaci)",
        "gejala": "Pucuk daun bronzing (berwarna perunggu kecokelatan), muncul bintik dan cincin nekrotik gelap konsentris, ujung tunas mati (dieback).",
        "penanganan": [
            "Pasang perangkap perekat biru atau kuning untuk memonitor populasi Trips.",
            "Semprotkan insektisida khusus Trips seperti Spinetoram atau Abamektin.",
            "Musnahkan tanaman yang terinfeksi akut agar tidak menularkan ke seluruh kebun."
        ]
    },
    "healthy": {
        "nama_id": "Tanaman Sehat (Bebas Gejala Penyakit)",
        "penyebab": "Tidak ditemukan patogen aktif yang merusak.",
        "gejala": "Permukaan daun hijau segar, turgor daun optimal, tidak terdapat bercak nekrotik, klorosis, atau deformasi struktur helai daun.",
        "penanganan": [
            "Lanjutkan pemeliharaan rutin dengan pemupukan berimbang NPK sesuai fase tumbuh.",
            "Lakukan penyiraman secukupnya di pangkal tanaman (hindari tanah terlalu becek).",
            "Lakukan monitoring berkala seminggu sekali untuk deteksi dini hama dan penyakit."
        ]
    }
}


def get_disease_details(label: str) -> dict:
    """Mengembalikan narasi lengkap gejala & rekomendasi penanganan berdasarkan label penyakit."""
    clean = re.sub(r"[^a-z0-9]", "", str(label).lower())

    # 1. Pencocokan langsung / sinonim
    if "healthy" in clean:
        return DISEASE_DB["healthy"]
    if "spottedwilt" in clean:
        return DISEASE_DB["spottedwilt"]
    if "leafcurl" in clean:
        return DISEASE_DB["leafcurl"]
    if "tomatobacterial" in clean or "bacterialspot" in clean:
        return DISEASE_DB["tomatobacterialspot"]
    if "septoria" in clean:
        return DISEASE_DB["septoria"]
    if "fleabeetle" in clean:
        return DISEASE_DB["fleabeetle"]
    if "mosaic" in clean:
        return DISEASE_DB["mosaic"]
    if "rust" in clean:
        return DISEASE_DB["rust"]
    if "blackrot" in clean:
        return DISEASE_DB["blackrot"]
    if "rot" in clean:
        return DISEASE_DB["leafrot"]
    if "cercospora" in clean or "cercopora" in clean:
        return DISEASE_DB["cercospora"]
    if "begomovirus" in clean:
        return DISEASE_DB["begomovirus"]
    if "verticillium" in clean:
        return DISEASE_DB["verticillium"]
    if "fusarium" in clean:
        return DISEASE_DB["fusarium"]
    if "bacterialwilt" in clean:
        return DISEASE_DB["bacterialwilt"]
    if "downy" in clean or "downey" in clean:
        return DISEASE_DB["downymildew"]
    if "blight" in clean:
        return DISEASE_DB["blight"]
    if "anthracnose" in clean:
        return DISEASE_DB["anthracnose"]
    if "spot" in clean:
        return DISEASE_DB["leafspot"]

    # Fallback jika ada label yang tidak terdaftar
    return {
        "nama_id": label,
        "penyebab": "Patogen tanaman (Jamur / Bakteri / Virus)",
        "gejala": "Terlihat perubahan pigmentasi warna, bercak nekrotik, atau kelainan fisiologis pada helai daun.",
        "penanganan": [
            "Pangkas daun yang bergejala sakit dan buang jauh dari tanaman sehat.",
            "Kurangi kelembapan di sekitar tanaman dengan menjaga jarak tanam.",
            "Gunakan pestisida hayati (seperti ekstrak daun mimba) atau konsultasikan ke penyuluh pertanian setempat."
        ]
    }
