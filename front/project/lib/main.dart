import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: LoginScreen(),
    );
  }
}

class LoginScreen extends StatelessWidget {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.blue[50],
      body: Center(
        child: Padding(
          padding: EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                "Login",
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue[800],
                ),
              ),
              SizedBox(height: 16.0),
              TextField(
                controller: _emailController,
                decoration: InputDecoration(
                  labelText: 'Email',
                  prefixIcon: Icon(Icons.email, color: Colors.blue[800]),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12.0),
                  ),
                ),
                keyboardType: TextInputType.emailAddress,
              ),
              SizedBox(height: 16.0),
              TextField(
                controller: _passwordController,
                decoration: InputDecoration(
                  labelText: 'Password',
                  prefixIcon: Icon(Icons.lock, color: Colors.blue[800]),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12.0),
                  ),
                ),
                obscureText: true,
              ),
              SizedBox(height: 24.0),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(vertical: 16.0),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12.0),
                  ),
                  backgroundColor: Colors.blue[800],
                ),
                onPressed: () {
                  String email = _emailController.text;
                  String password = _passwordController.text;
                  print('Email: $email, Password: $password');
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => HomeScreen()),
                  );
                },
                child: Center(
                  child: Text(
                    'Login',
                    style: TextStyle(
                      fontSize: 18,
                      color: Colors.white,
                    ),
                  ),
                ),
              ),
              SizedBox(height: 16.0),
              TextButton(
                onPressed: () {
                  // Logica de "Esqueceu a senha?"
                },
                child: Text(
                  "Esqueceu a senha?",
                  style: TextStyle(color: Colors.blue[800]),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  final List<Map<String, dynamic>> orders = [
    {
      'id': '1354a32',
      'description': 'Troca de óleo',
      'equipment': 'WEG W22',
      'status': 'ABERTA',
      'statusColor': 'red',
      'tools': [
        'MAT901 (Chave de Fenda)',
        'MAT903 (Martelo)',
        'MAT904 (Torquímetro)',
        'MAT906 (Chave Estrela)',
        'MAT302 (Óleo 10W30)',
      ],
      'imageUrl': 'https://bimgix.tractian.com/motor-eletrico.png',
    },
    {
      'id': '1354a32',
      'description': 'Troca de óleo',
      'equipment': 'WEG W22',
      'status': 'ABERTA',
      'statusColor': 'red',
      'tools': [
        'MAT901 (Chave de Fenda)',
        'MAT903 (Martelo)',
        'MAT904 (Torquímetro)',
        'MAT906 (Chave Estrela)',
        'MAT302 (Óleo 10W30)',
      ],
      'imageUrl': 'https://example.com/motor-image.jpg',
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.blue[50],
      appBar: AppBar(
        backgroundColor: Colors.blue[800],
        title: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('José Paulo', style: TextStyle(color: Colors.white)),
            CircleAvatar(
              backgroundImage: NetworkImage(
                  'https://example.com/user-profile.jpg'),
            ),
          ],
        ),
      ),
      body: ListView.builder(
        padding: EdgeInsets.all(16.0),
        itemCount: orders.length,
        itemBuilder: (context, index) {
          final order = orders[index];
          return GestureDetector(
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => OrderDetailScreen(order: order),
                ),
              );
            },
            child: Card(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16.0),
              ),
              margin: EdgeInsets.symmetric(vertical: 8.0),
              color: Colors.white,
              child: Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Ordem de Serviço ${order['id']}',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                        color: Colors.blue[800],
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      order['description'],
                      style: TextStyle(fontSize: 16, color: Colors.grey[800]),
                    ),
                    Text(
                      order['equipment'],
                      style: TextStyle(fontSize: 14, color: Colors.grey[600]),
                    ),
                    SizedBox(height: 8),
                    Align(
                      alignment: Alignment.bottomRight,
                      child: Text(
                        order['status'],
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: order['statusColor'] == 'red'
                              ? Colors.red
                              : order['statusColor'] == 'orange'
                                  ? Colors.orange
                                  : Colors.green,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class OrderDetailScreen extends StatelessWidget {
  final Map<String, dynamic> order;

  OrderDetailScreen({required this.order});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.blue[50],
      appBar: AppBar(
        backgroundColor: Colors.blue[800],
        title: Text(
          order['description'],
          style: TextStyle(color: Colors.white),
        ),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              order['description'],
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 28,
                color: Colors.blue[800],
              ),
            ),
            Text(
              order['equipment'],
              style: TextStyle(
                fontSize: 18,
                color: Colors.grey[700],
              ),
            ),
            SizedBox(height: 16),
            ClipRRect(
              borderRadius: BorderRadius.circular(12.0),
              child: Container(
                height: 150, // Defina a altura desejada
                width: double.infinity, // Ajuste à largura total
                child: FittedBox(
                  fit: BoxFit.contain, // Use BoxFit.contain se desejar ver a imagem completa
                  child: Image.network(
                    order['imageUrl'],
                    width: 1080, // Largura original da imagem
                    height: 1080, // Altura original da imagem
                  ),
                ),
              ),
            ),
            SizedBox(height: 16),
            Text(
              'Ferramentas Necessárias',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 20,
                color: Colors.blue[800],
              ),
            ),
            SizedBox(height: 8),
            ...order['tools'].map<Widget>((tool) => Padding(
                  padding: EdgeInsets.symmetric(vertical: 4.0),
                  child: Text(
                    tool,
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.grey[800],
                    ),
                  ),
                )),
            Spacer(),
            Center(
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(vertical: 16.0, horizontal: 24.0),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12.0),
                  ),
                  backgroundColor: Colors.blue[800],
                ),
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => StepByStepGuide(),
                    ),
                  );
                },
                child: Text(
                  'Iniciar Guia',
                  style: TextStyle(fontSize: 18, color: Colors.white),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class StepByStepGuide extends StatefulWidget {
  @override
  _StepByStepGuideState createState() => _StepByStepGuideState();
}

class _StepByStepGuideState extends State<StepByStepGuide> {
  final PageController _pageController = PageController();
  final ImagePicker _picker = ImagePicker();
  File? _capturedImage;
  
  Uint8List? _returnedImageBytes; // Armazena bytes da imagem retornada
  bool _isUploading = false;
  int _currentPage = 0;

  final List<Map<String, dynamic>> steps = [
    {'title': 'Passo 1', 'description': 'Remover plug do Dreno', 'type': 'camera', 'object': 'all'},
    {'title': 'Passo 2', 'description': 'Desconectar mangueira', 'type': 'image', 'imageUrl': 'https://example.com/image2.jpg'},
    {'title': 'Passo 3', 'description': 'Limpar o filtro de óleo', 'type': 'text', 'textContent': 'Certifique-se de limpar o filtro de óleo adequadamente para evitar obstruções.'},
  ];

  Future<void> _takePicture(String object) async {
    final pickedFile = await _picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      setState(() {
        _capturedImage = File(pickedFile.path);
        _returnedImageBytes = null; // Limpa imagem retornada anterior, se houver
      });
      await _uploadImage(object); // Envia a imagem após capturá-la
    }
  }

  Future<void> _uploadImage(String object) async {
    if (_capturedImage == null) return;

    setState(() {
      _isUploading = true;
    });

    final url = Uri.parse("https://1092-2801-b0-20-59-1831-d5a3-3faa-aca2.ngrok-free.app/imagem/$object");

    try {
      var request = http.MultipartRequest("POST", url);
      request.files.add(await http.MultipartFile.fromPath('image', _capturedImage!.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        final imageBytes = await response.stream.toBytes();
        setState(() {
          _capturedImage = null; // Limpa a imagem original para mostrar a processada
          _returnedImageBytes = imageBytes; // Armazena os bytes da imagem retornada
        });
        _showSnackBar("Imagem enviada com sucesso!");
      } else {
        _showSnackBar("Erro ao enviar a imagem. Status: ${response.statusCode}");
      }
    } catch (e) {
      _showSnackBar("Erro: $e");
    } finally {
      setState(() {
        _isUploading = false;
      });
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  Widget _buildStepContent(Map<String, dynamic> step) {
    if (step['type'] == 'camera') {
      return Column(
        children: [
          _returnedImageBytes != null
              ? ClipRRect(
                  borderRadius: BorderRadius.circular(12.0),
                  child: Image.memory(
                    _returnedImageBytes!,
                    height: 300,
                    width: double.infinity,
                    fit: BoxFit.cover,
                  ),
                )
              : _capturedImage != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(12.0),
                      child: Image.file(
                        _capturedImage!,
                        height: 300,
                        width: double.infinity,
                        fit: BoxFit.cover,
                      ),
                    )
                  : Container(
                      height: 300,
                      width: double.infinity,
                      color: Colors.grey[300],
                      child: Center(
                        child: Text("Nenhuma imagem capturada"),
                      ),
                    ),
          SizedBox(height: 16),
          ElevatedButton.icon(
            icon: Icon(Icons.camera),
            label: Text("Capturar Imagem"),
            onPressed: () => _takePicture(step['object']),
            style: ElevatedButton.styleFrom(
              padding: EdgeInsets.symmetric(vertical: 16.0),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12.0),
              ),
              backgroundColor: Colors.blue[800],
            ),
          ),
        ],
      );
    } else if (step['type'] == 'image') {
      return ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Image.network(
          step['imageUrl'],
          height: 200,
          width: double.infinity,
          fit: BoxFit.cover,
        ),
      );
    } else if (step['type'] == 'text') {
      return Container(
        padding: EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.blue[800]!, width: 1.5),
        ),
        child: Text(
          step['textContent'],
          style: TextStyle(fontSize: 16, color: Colors.grey[800]),
          textAlign: TextAlign.center,
        ),
      );
    }
    return SizedBox.shrink();
  }

  void _nextPage() {
    if (_currentPage < steps.length - 1) {
      _pageController.nextPage(
        duration: Duration(milliseconds: 300),
        curve: Curves.easeIn,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.blue[50],
      appBar: AppBar(
        backgroundColor: Colors.blue[800],
        title: Text(
          'Guia de Manutenção',
          style: TextStyle(color: Colors.white),
        ),
      ),
      body: Stack(
        children: [
          PageView.builder(
            controller: _pageController,
            onPageChanged: (index) {
              setState(() {
                _currentPage = index;
                _capturedImage = null; // Reseta a imagem capturada ao mudar de passo
                _returnedImageBytes = null; // Reseta a imagem retornada ao mudar de passo
              });
            },
            itemCount: steps.length,
            itemBuilder: (context, index) {
              final step = steps[index];
              return Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      step['title'],
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 24,
                        color: Colors.blue[800],
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      step['description'],
                      style: TextStyle(
                        fontSize: 18,
                        color: Colors.grey[700],
                      ),
                    ),
                    SizedBox(height: 16),
                    _buildStepContent(step),
                    Spacer(),
                    Center(
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          padding: EdgeInsets.symmetric(vertical: 16.0),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12.0),
                          ),
                          backgroundColor: Colors.blue[800],
                        ),
                        onPressed: _nextPage,
                        child: Text(
                          _currentPage < steps.length - 1 ? 'Próximo' : 'Concluir',
                          style: TextStyle(fontSize: 18, color: Colors.white),
                        ),
                      ),
                    ),
                  ],
                ),
              );
            },
          ),
          if (_isUploading)
            Center(
              child: Container(
                color: Colors.black54,
                child: CircularProgressIndicator(
                  color: Colors.white,
                ),
              ),
            ),
        ],
      ),
    );
  }
}