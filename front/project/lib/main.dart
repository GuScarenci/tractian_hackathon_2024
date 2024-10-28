import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
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
      appBar: AppBar(
        title: Text('Login'),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: _emailController,
              decoration: InputDecoration(
                labelText: 'Email',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.emailAddress,
            ),
            SizedBox(height: 16.0),
            TextField(
              controller: _passwordController,
              decoration: InputDecoration(
                labelText: 'Password',
                border: OutlineInputBorder(),
              ),
              obscureText: true,
            ),
            SizedBox(height: 16.0),
            ElevatedButton(
              onPressed: () {
                String email = _emailController.text;
                String password = _passwordController.text;
                print('Email: $email, Password: $password');

                // Navegar para a próxima página
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => HomeScreen()),
                );
              },
              child: Text('Login'),
            ),
          ],
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
      'imageUrl': 'https://example.com/motor-image.jpg',
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('José Paulo'),
            CircleAvatar(
              backgroundImage: NetworkImage(
                  'https://example.com/user-profile.jpg'),
            ),
          ],
        ),
      ),
      body: ListView.builder(
        padding: EdgeInsets.all(8.0),
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
                borderRadius: BorderRadius.circular(12),
                side: BorderSide(color: Colors.blue, width: 2),
              ),
              color: Colors.blue[50],
              margin: EdgeInsets.symmetric(vertical: 8.0),
              child: Padding(
                padding: EdgeInsets.all(12.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Ordem de Serviço ${order['id']}',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      order['description'],
                      style: TextStyle(fontSize: 14),
                    ),
                    Text(
                      order['equipment'],
                      style: TextStyle(fontSize: 14),
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
      appBar: AppBar(
        title: Text(order['description']),
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
                fontSize: 24,
              ),
            ),
            Text(
              order['equipment'],
              style: TextStyle(fontSize: 18),
            ),
            SizedBox(height: 16),
            Image.network(
              order['imageUrl'],
              height: 100,
            ),
            SizedBox(height: 16),
            Text(
              'FERRAMENTAS',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
              ),
            ),
            SizedBox(height: 8),
            ...order['tools'].map<Widget>((tool) => Text(
                  tool,
                  style: TextStyle(fontSize: 14),
                )),
            Spacer(),
            Center(
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => StepByStepGuide(),
                    ),
                  );
                },
                child: Text('Iniciar Guia'),
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
  int _currentPage = 0;

  final List<Map<String, dynamic>> steps = [
    {'title': 'Passo 1', 'description': 'Remover plug do Dreno', 'type': 'camera'},
    {'title': 'Passo 2', 'description': 'Desconectar mangueira', 'type': 'image', 'imageUrl': 'https://example.com/image2.jpg'},
    {'title': 'Passo 3', 'description': 'Limpar o filtro de óleo', 'type': 'text', 'textContent': 'Certifique-se de limpar o filtro de óleo adequadamente para evitar obstruções.'},
  ];

  void _nextPage() {
    if (_currentPage < steps.length - 1) {
      _pageController.nextPage(
        duration: Duration(milliseconds: 300),
        curve: Curves.easeIn,
      );
    }
  }

  Widget _buildStepContent(Map<String, dynamic> step) {
    switch (step['type']) {
      case 'camera':
        return Container(
          color: Colors.grey[300],
          height: 200,
          width: double.infinity,
          child: Center(
            child: Icon(
              Icons.camera_alt,
              size: 50,
              color: Colors.grey[700],
            ),
          ),
        );
      case 'image':
        return Image.network(
          step['imageUrl'],
          height: 200,
          width: double.infinity,
          fit: BoxFit.cover,
        );
      case 'text':
        return Text(
          step['textContent'],
          style: TextStyle(fontSize: 16),
          textAlign: TextAlign.center,
        );
      default:
        return SizedBox.shrink();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Guia de Manutenção'),
      ),
      body: PageView.builder(
        controller: _pageController,
        onPageChanged: (index) {
          setState(() {
            _currentPage = index;
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
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  step['description'],
                  style: TextStyle(fontSize: 18),
                ),
                SizedBox(height: 16),
                _buildStepContent(step),
                Spacer(),
                Center(
                  child: ElevatedButton(
                    onPressed: _nextPage,
                    child: Text(_currentPage < steps.length - 1
                        ? 'Próximo'
                        : 'Concluir'),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}